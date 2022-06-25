import logging
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from clai.modeling.module.head import base
from clai.tooling import block
from clai.training.module import loss

logger = logging.getLogger(__name__)


class TextClassificationHead(base.PredictionHead):
    def __init__(
        self,
        layer_dims=None,
        num_labels=None,
        class_weights=None,
        loss_ignore_index=-100,
        loss_reduction="none",
        task_name="text_classification",
        label_list=None,
    ):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_ignore_index:
        :param loss_reduction:
        :param task_name:
        :param kwargs:
        """
        super(TextClassificationHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning("`layer_dims` will be deprecated in future releases")
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        self.feed_forward = block.FeedForwardBlock(self.layer_dims)
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_sequence"
        self.model_type = "text_classification"
        self.task_name = task_name  # used for connecting with the right output of the processor

        if type(class_weights) is np.ndarray and class_weights.ndim != 1:
            raise ValueError(
                "When you pass `class_weights` as `np.ndarray` it must have 1 dimension! "
                "You provided {} dimensions.".format(class_weights.ndim)
            )

        self.class_weights = class_weights

        if class_weights is not None:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            balanced_weights = None

        self.loss_fct = loss.CELoss(class_weights=balanced_weights, reduction=loss_reduction, ignore_index=loss_ignore_index)
        # self.loss_fct = nn.CrossEntropyLoss(
        #     weight=balanced_weights,
        #     reduction=loss_reduction,
        #     ignore_index=loss_ignore_index,
        # )

        # add label list
        if label_list:
            self.label_list = label_list

        self.generate_config()

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path,
        revision=None,
        layer_dims=None,
        num_labels=None,
        class_weights=None,
        loss_ignore_index=-100,
        loss_reduction="none",
        task_name="text_classification",
        label_list=None,
    ):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public name:
                                              - deepset/bert-base-german-cased-hatespeech-GermEval18Coarse

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """

        if (
            os.path.exists(pretrained_model_name_or_path)
            and "config.json" in pretrained_model_name_or_path
            and "prediction_head" in pretrained_model_name_or_path
        ):
            # a) FARM style
            head = super(TextClassificationHead, cls)._load(pretrained_model_name_or_path)
        else:
            # b) transformers style
            # load all weights from model
            full_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, revision=revision)
            # init empty head
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.id2label)])
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            # add label list
            head.label_list = list(full_model.config.id2label.values())
            del full_model

        return head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids
        return self.loss_fct(logits, label_ids.view(-1))

    def logits_to_probs(self, logits, return_class_probs, **kwargs):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        if return_class_probs:
            probs = probs
        else:
            probs = torch.max(probs, dim=1)[0]
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, **kwargs):
        logits = logits.cpu().numpy()
        pred_ids = logits.argmax(1)
        preds = [self.label_list[int(x)] for x in pred_ids]
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        # This is the standard doc classification case
        try:
            labels = [self.label_list[int(x)] for x in label_ids]
        # This case is triggered in Natural Questions where each example can have multiple labels
        except TypeError:
            labels = [self.label_list[int(x[0])] for x in label_ids]
        return labels

    def formatted_preds(self, logits=None, preds=None, samples=None, return_class_probs=False, **kwargs):
        """Like QuestionAnsweringHead.formatted_preds(), this fn can operate on either logits or preds. This
        is needed since at inference, the order of operations is very different depending on whether we are performing
        aggregation or not (compare Inferencer._get_predictions() vs Inferencer._get_predictions_and_aggregate())"""

        assert (logits is not None) or (preds is not None)

        # When this method is used along side a QAHead at inference (e.g. Natural Questions), preds is the input and
        # there is currently no good way of generating probs
        if logits is not None:
            preds = self.logits_to_preds(logits)
            probs = self.logits_to_probs(logits, return_class_probs)
        else:
            probs = [None] * len(preds)

        # TODO this block has to do with the difference in Basket and Sample structure between SQuAD and NQ
        try:
            contexts = [sample.clear_text["text"] for sample in samples]
        # This case covers Natural Questions where the sample is in a QA style
        except KeyError:
            contexts = [sample.clear_text["question_text"] + " | " + sample.clear_text["passage_text"] for sample in samples]

        contexts_b = [sample.clear_text["text_b"] for sample in samples if "text_b" in sample.clear_text]
        if len(contexts_b) != 0:
            contexts = ["|".join([a, b]) for a, b in zip(contexts, contexts_b)]

        res = {"task": "text_classification", "predictions": []}
        for pred, prob, context in zip(preds, probs, contexts):
            if not return_class_probs:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": f"{pred}",
                    "probability": prob,
                }
            else:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": "class_probabilities",
                    "probability": prob,
                }

            res["predictions"].append(pred_dict)
        return res
