import io
import pathlib

import numpy as np
import torch
import transformers

from clai.tooling import nlp


class LanguageModel(torch.nn.Module):
    """
    The main class wrapping different huggingface model(s). The main reason is "HF" forward pass require different params for various LM.
    E.g. Distilled BERT doesn't have segment_ids
    """

    subclasses = {}

    def __init_subclass__(cls, calling_name: str = None, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific Formatter implementation.
        """
        super().__init_subclass__(**kwargs)
        calling_name = cls.__name__ if calling_name is None else calling_name
        cls.subclasses[calling_name] = cls

    @classmethod
    def load(cls, name: str = None, **kwargs):
        klass = cls.subclasses[name] if name is not None else cls.subclasses[cls.__name__]
        return klass.load(**kwargs)

    def freeze(self, layers):
        """To be implemented"""
        raise NotImplementedError()

    def unfreeze(self):
        """To be implemented"""
        raise NotImplementedError()

    def save_config(self, save_dir):
        save_filename = pathlib.Path(save_dir) / "language_model_config.json"
        with io.open(save_filename, "w") as f:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            # For DPR models, transformers overwrites the model_type with the one set in DPRConfig
            # Therefore, we copy the model_type from the model config to DPRConfig
            if self.__class__.__name__ == "DPRQuestionEncoder" or self.__class__.__name__ == "DPRContextEncoder":
                setattr(transformers.DPRConfig, "model_type", self.model.config.model_type)
            string = self.model.config.to_json_string()
            f.write(string)

    def save(self, save_dir, state_dict=None):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        :param state_dict: A dictionary containing a whole state of the module including names of layers. By default, the unchanged state dict of the module is used
        :type state_dict: dict
        """
        # Save Weights
        save_name = pathlib.Path(save_dir) / "language_model.bin"
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # Only save the model it-self

        if not state_dict:
            state_dict = model_to_save.state_dict()
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False

    def formatted_preds(self, logits, samples, ignore_first_token=True, padding_mask=None, input_ids=None, **kwargs):
        """
        Extracting vectors from language model (e.g. for extracting sentence embeddings).
        Different pooling strategies and layers are available and will be determined from the object attributes
        `extraction_layer` and `extraction_strategy`. Both should be set via the Inferencer:
        Example:  Inferencer(extraction_strategy='cls_token', extraction_layer=-1)

        :param logits: Tuple of (sequence_output, pooled_output) from the language model.
                       Sequence_output: one vector per token, pooled_output: one vector for whole sequence
        :param samples: For each item in logits we need additional meta information to format the prediction (e.g. input text).
                        This is created by the Processor and passed in here from the Inferencer.
        :param ignore_first_token: Whether to include the first token for pooling operations (e.g. reduce_mean).
                                   Many models have here a special token like [CLS] that you don't want to include into your average of token embeddings.
        :param padding_mask: Mask for the padding tokens. Those will also not be included in the pooling operations to prevent a bias by the number of padding tokens.
        :param input_ids: ids of the tokens in the vocab
        :param kwargs: kwargs
        :return: list of dicts containing preds, e.g. [{"context": "some text", "vec": [-0.01, 0.5 ...]}]
        """

        if not hasattr(self, "extraction_layer") or not hasattr(self, "extraction_strategy"):
            raise ValueError(
                "`extraction_layer` or `extraction_strategy` not specified for LM. "
                "Make sure to set both, e.g. via Inferencer(extraction_strategy='cls_token', extraction_layer=-1)`"
            )

        # unpack the tuple from LM forward pass
        sequence_output = logits[0][0]
        pooled_output = logits[0][1]

        # aggregate vectors
        if self.extraction_strategy == "pooled":
            if self.extraction_layer != -1:
                raise ValueError(
                    f"Pooled output only works for the last layer, but got extraction_layer = {self.extraction_layer}. Please set `extraction_layer=-1`.)"
                )
            vecs = pooled_output.cpu().numpy()
        elif self.extraction_strategy == "per_token":
            vecs = sequence_output.cpu().numpy()
        elif self.extraction_strategy == "reduce_mean":
            vecs = self._pool_tokens(
                sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token
            )
        elif self.extraction_strategy == "reduce_max":
            vecs = self._pool_tokens(
                sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token
            )
        elif self.extraction_strategy == "cls_token":
            vecs = sequence_output[:, 0, :].cpu().numpy()
        elif self.extraction_strategy == "s3e":
            vecs = self._pool_tokens(
                sequence_output,
                padding_mask,
                self.extraction_strategy,
                ignore_first_token=ignore_first_token,
                input_ids=input_ids,
                s3e_stats=self.s3e_stats,
            )
        else:
            raise NotImplementedError

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.clear_text["text"]
            pred["vec"] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(self, sequence_output, padding_mask, strategy, ignore_first_token, input_ids=None, s3e_stats=None):

        token_vecs = sequence_output.cpu().numpy()
        # we only take the aggregated value of non-padding tokens
        padding_mask = padding_mask.cpu().numpy()
        ignore_mask_2d = padding_mask == 0
        # sometimes we want to exclude the CLS token as well from our aggregation operation
        if ignore_first_token:
            ignore_mask_2d[:, 0] = True
        ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
        ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, np.newaxis]
        if strategy == "reduce_max":
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).max(axis=1).data
        if strategy == "reduce_mean":
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).mean(axis=1).data
        if strategy == "s3e":
            input_ids = input_ids.cpu().numpy()
            pooled_vecs = nlp.s3e_pooling(
                token_embs=token_vecs,
                token_ids=input_ids,
                token_weights=s3e_stats["token_weights"],
                centroids=s3e_stats["centroids"],
                token_to_cluster=s3e_stats["token_to_cluster"],
                svd_components=s3e_stats.get("svd_components", None),
                mask=padding_mask == 0,
            )
        return pooled_vecs


class ImageModel(torch.nn.Module):

    subclasses = {}

    def __init_subclass__(cls, calling_name: str = None, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific Formatter implementation.
        """
        super().__init_subclass__(**kwargs)
        calling_name = cls.__name__ if calling_name is None else calling_name
        cls.subclasses[calling_name] = cls


class TTSModel(torch.nn.Module):

    subclasses = {}

    def __init_subclass__(cls, calling_name: str = None, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific Formatter implementation.
        """
        super().__init_subclass__(**kwargs)
        calling_name = cls.__name__ if calling_name is None else calling_name
        cls.subclasses[calling_name] = cls
