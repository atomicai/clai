import io
import logging
import os
import pathlib

import numpy as np
import simplejson
import torch
import torch.nn as nn

from clai.tooling import block, tool

logger = logging.getLogger(__name__)


class PredictionHead(nn.Module):
    """Takes word embeddings from a language model and generates logits for a given task. Can also convert logits
    to loss and and logits to predictions."""

    subclasses = {}

    def __init_subclass__(cls, calling_name: str = None, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific Formatter implementation.
        """
        super().__init_subclass__(**kwargs)
        calling_name = cls.__name__ if calling_name is None else calling_name
        cls.subclasses[calling_name] = cls

    @classmethod
    def create(cls, prediction_head_name, layer_dims, class_weights=None, **kwargs):
        """
        Create subclass of Prediction Head.

        :param prediction_head_name: Classname (exact string!) of prediction head we want to create
        :type prediction_head_name: str
        :param layer_dims: describing the feed forward block structure, e.g. [768,2]
        :type layer_dims: List[Int]
        :param class_weights: The loss weighting to be assigned to certain label classes during training.
           Used to correct cases where there is a strong class imbalance.
        :type class_weights: list[Float]
        :return: Prediction Head of class prediction_head_name
        """
        # TODO make we want to make this more generic.
        #  1. Class weights is not relevant for all heads.
        #  2. Layer weights impose FF structure, maybe we want sth else later
        # Solution: We could again use **kwargs
        return cls.subclasses[prediction_head_name](layer_dims=layer_dims, class_weights=class_weights, **kwargs)

    def save_config(self, save_dir, head_num=0):
        """
        Saves the config as a json file.

        :param save_dir: Path to save config to
        :type save_dir: str or Path
        :param head_num: Which head to save
        :type head_num: int
        """
        # updating config in case the parameters have been changed
        self.generate_config()
        output_config_file = pathlib.Path(save_dir) / f"prediction_head_{head_num}_config.json"
        with io.open(output_config_file, "w") as f:
            simplejson.dump(self.config, f)

    def save(self, save_dir, head_num=0):
        """
        Saves the prediction head state dict.

        :param save_dir: path to save prediction head to
        :type save_dir: str or Path
        :param head_num: which head to save
        :type head_num: int
        """
        output_model_file = pathlib.Path(save_dir) / f"prediction_head_{head_num}.bin"
        torch.save(self.state_dict(), output_model_file)
        self.save_config(save_dir, head_num)

    def generate_config(self):
        """
        Generates config file from Class parameters (only for sensible config parameters).
        """
        config = {}
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                value = value.tolist()
            if tool.is_json(value) and key[0] != "_":
                config[key] = value
            if self.task_name == "text_similarity" and key == "similarity_function":
                config['similarity_function'] = value
        config["name"] = self.__class__.__name__
        config.pop("config", None)
        self.config = config

    @classmethod
    def load(cls, name: str = None, **kwargs):
        klass = cls.subclasses[name] if name is not None else cls.subclasses[cls.__name__]
        return klass.load(**kwargs)

    @classmethod
    def _load(cls, config_file, strict=True, load_weights=True):
        """
        Loads a Prediction Head. Infers the class of prediction head from config_file.

        :param config_file: location where corresponding config is stored
        :type config_file: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :return: PredictionHead
        :rtype: PredictionHead[T]
        """
        config = simplejson.load(io.open(config_file))
        prediction_head = cls.subclasses[config["name"]](**config)
        if load_weights:
            model_file = cls._get_model_file(config_file=config_file)
            logger.info("Loading prediction head from {}".format(model_file))
            prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")), strict=strict)
        return prediction_head

    def logits_to_loss(self, logits, labels):
        """
        Implement this function in your special Prediction Head.
        Should combine logits and labels with a loss fct to a per sample loss.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param labels: labels, can vary in shape and type, depending on task
        :type labels: object
        :return: per sample loss as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def logits_to_preds(self, logits):
        """
        Implement this function in your special Prediction Head.
        Should combine turn logits into predictions.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :return: predictions as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def prepare_labels(self, **kwargs):
        """
        Some prediction heads need additional label conversion.
        E.g. NER needs word level labels turned into subword token level labels.

        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: labels in the right format
        :rtype: object
        """
        # TODO maybe just return **kwargs to not force people to implement this
        raise NotImplementedError()

    def resize_input(self, input_dim):
        """This function compares the output dimensionality of the language model against the input dimensionality
        of the prediction head. If there is a mismatch, the prediction head will be resized to fit."""
        if "feed_forward" not in dir(self):
            return
        else:
            old_dims = self.feed_forward.layer_dims
            if input_dim == old_dims[0]:
                return
            new_dims = [input_dim] + old_dims[1:]
            logger.info(
                f"Resizing input dimensions of {type(self).__name__} ({self.task_name}) "
                f"from {old_dims} to {new_dims} to match language model"
            )
            self.feed_forward = block.FeedForwardBlock(new_dims)
            self.layer_dims[0] = input_dim
            self.feed_forward.layer_dims[0] = input_dim

    @classmethod
    def _get_model_file(cls, config_file):
        if "config.json" in str(config_file) and "prediction_head" in str(config_file):
            head_num = int("".join([char for char in os.path.basename(config_file) if char.isdigit()]))
            model_file = pathlib.Path(os.path.dirname(config_file)) / f"prediction_head_{head_num}.bin"
        else:
            raise ValueError(f"This doesn't seem to be a proper prediction_head config file: '{config_file}'")
        return model_file

    def _set_name(self, name):
        self.task_name = name
