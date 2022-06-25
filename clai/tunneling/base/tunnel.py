import logging
import os
import pathlib

import torch.nn as nn

logger = logging.getLogger(__name__)


class Tunnel:
    """
    Base Class for implementing AdaptiveModel with frameworks like PyTorch and ONNX.
    """

    subclasses = {}

    def __init_subclass__(cls, calling_name: str = None, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific `Tunnel` implementation.
        """
        super().__init_subclass__(**kwargs)
        calling_name = cls.__name__ if calling_name is None else calling_name
        cls.subclasses[calling_name] = cls

    def __init__(self, prediction_heads):
        self.prediction_heads = prediction_heads

    @classmethod
    def load(cls, **kwargs):
        """
        Load corresponding AdaptiveModel Class(AdaptiveModel/ONNXAdaptiveModel) based on the
        files in the load_dir.

        :param kwargs: arguments to pass for loading the model.
        :return: instance of a model
        """
        if (pathlib.Path(kwargs["load_dir"]) / "model.onnx").is_file():
            model = cls.subclasses["ONNXAdaptiveModel"].load(**kwargs)
        else:
            model = cls.subclasses["AdaptiveModel"].load(**kwargs)
        return model

    def logits_to_preds(self, logits, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param label_maps: Maps from label encoding to label string
        :param label_maps: dict
        :return: A list of all predictions from all prediction heads
        """
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def formatted_preds(self, logits, **kwargs):
        """
        Format predictions for inference.

        :param logits: model logits
        :type logits: torch.tensor
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: predictions in the right format
        """
        n_heads = len(self.prediction_heads)

        if n_heads == 0:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            preds_final = self.language_model.formatted_preds(logits=logits, **kwargs)

        elif n_heads == 1:
            preds_final = []
            # This try catch is to deal with the fact that sometimes we collect preds before passing it to
            # formatted_preds (see Inferencer._get_predictions_and_aggregate()) and sometimes we don't
            # (see Inferencer._get_predictions())
            try:
                preds = kwargs["preds"]
                temp = [y[0] for y in preds]
                preds_flat = [item for sublist in temp for item in sublist]
                kwargs["preds"] = preds_flat
            except KeyError:
                kwargs["preds"] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)
            # TODO This is very messy - we need better definition of what the output should look like
            if type(preds) == list:
                preds_final += preds
            elif type(preds) == dict and "predictions" in preds:
                preds_final.append(preds)

        # This case is triggered by Natural Questions
        else:
            preds_final = [list() for _ in range(n_heads)]
            preds = kwargs["preds"]
            preds_for_heads = stack(preds)
            logits_for_heads = [None] * n_heads

            samples = [s for b in kwargs["baskets"] for s in b.samples]
            kwargs["samples"] = samples

            del kwargs["preds"]

            for i, (head, preds_for_head, logits_for_head) in enumerate(
                zip(self.prediction_heads, preds_for_heads, logits_for_heads)
            ):
                preds = head.formatted_preds(logits=logits_for_head, preds=preds_for_head, **kwargs)
                preds_final[i].append(preds)

            # Look for a merge() function amongst the heads and if a single one exists, apply it to preds_final
            merge_fn = pick_single_fn(self.prediction_heads, "merge_formatted_preds")
            if merge_fn:
                preds_final = merge_fn(preds_final)

        return preds_final

    def connect_heads_with_processor(self, tasks, require_labels=True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels)
        :return:
        """

        # Drop the next sentence prediction head if it does not appear in tasks. This is triggered by the interaction
        # setting the argument BertStyleLMProcessor(next_sent_pred=False)
        if "nextsentence" not in tasks:
            idx = None
            for i, ph in enumerate(self.prediction_heads):
                if ph.task_name == "nextsentence":
                    idx = i
            if idx is not None:
                logger.info("Removing the NextSentenceHead since next_sent_pred is set to False in the BertStyleLMProcessor")
                del self.prediction_heads[i]

        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task \'{head.task_name}\' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            if "RegressionHead" in str(type(head)):
                # This needs to be explicitly set because the regression label_list is being hijacked to store
                # the scaling factor and the mean
                pass
            else:
                len(label_list)
            head.metric = tasks[head.task_name]["metric"]

    @classmethod
    def _get_prediction_head_files(cls, load_dir, strict=True):
        load_dir = pathlib.Path(load_dir)
        files = os.listdir(load_dir)
        model_files = [load_dir / f for f in files if ".bin" in f and "prediction_head" in f]
        config_files = [load_dir / f for f in files if "config.json" in f and "prediction_head" in f]
        # sort them to get correct order in case of multiple prediction heads
        model_files.sort()
        config_files.sort()

        if strict:
            error_str = (
                f"There is a mismatch in number of model files ({len(model_files)}) and config files ({len(config_files)})."
                "This might be because the Language Model Prediction Head "
                "does not currently support saving and loading"
            )
            assert len(model_files) == len(config_files), error_str
        logger.info(f"Found files for loading {len(model_files)} prediction heads")

        return model_files, config_files
