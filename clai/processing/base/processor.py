import abc
import io
import logging
import os
import pathlib
import random
from typing import Dict, List

import simplejson

from clai.tooling.io import load_tokenizer

logger = logging.getLogger(__name__)


class Processor(abc.ABC):

    subclasses = {}

    def __init_subclass__(cls, calling_name: str = None, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific `Processor` implementation.
        """
        super().__init_subclass__(**kwargs)
        calling_name = cls.__name__ if calling_name is None else calling_name
        cls.subclasses[calling_name] = cls

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        train_filename,
        dev_filename,
        test_filename,
        dev_split,
        data_dir,
        tasks={},
        proxies=None,
        multithreading_rust=True,
        logger=None,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param data_dir: The directory in which the train, test and perhaps dev files can be found.
        :type data_dir: str
        :param tasks: Tasks for which the processor shall extract labels from the input data.
                      Usually this includes a single, default task, e.g. text classification.
                      In a multitask setting this includes multiple tasks, e.g. 2x text classification.
                      The task name will be used to connect with the related PredictionHead.
        :type tasks: dict
        :param proxies: proxy configuration to allow downloads of remote datasets.
                    Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param multithreading_rust: Whether to allow multithreading in Rust, e.g. for FastTokenizers.
                                    Note: Enabling multithreading in Rust AND multiprocessing in python might cause
                                    deadlocks.
        :type multithreading_rust: bool
        """
        if not multithreading_rust:
            os.environ["RAYON_RS_NUM_CPUS"] = "1"

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.proxies = proxies

        # data sets
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        if data_dir:
            self.data_dir = pathlib.Path(data_dir)
        else:
            self.data_dir = None
        self.baskets = []
        self.logger = logger
        self._log_params()
        self.problematic_sample_ids = set()

    @classmethod
    def load(cls, name: str, data_dir, tokenizer, max_seq_len, **kwargs):
        processor = cls.subclasses[name](
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        return processor

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Infers the specific type of Processor from a config file (e.g. GNADProcessor) and loads an instance of it.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of a Processor Subclass (e.g. GNADProcessor)
        """
        # read config
        processor_config_file = pathlib.Path(load_dir) / "processor_config.json"
        config = simplejson.load(io.open(processor_config_file))
        config["inference"] = True

        tokenizer = load_tokenizer(load_dir, tokenizer_class=config["tokenizer"])
        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name=config["processor"], **config)

        for task_name, task in config["tasks"].items():
            processor.add_task(
                name=task_name,
                metric=task["metric"],
                label_list=task["label_list"],
                label_column_name=task["label_column_name"],
                text_column_name=task.get("text_column_name", None),
                task_type=task["task_type"],
            )

        if processor is None:
            raise Exception

        return processor

    @abc.abstractmethod
    def dataset_from_dicts(self, dicts: List[Dict]):
        pass

    @abc.abstractmethod
    def preprocess(self, txt: str, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def file_to_dicts(self, f: str) -> [dict]:
        pass

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = pathlib.Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as f:
            simplejson.dump(config, f)

    def add_task(self, name, metric, label_list, label_column_name=None, label_name=None, task_type=None, text_column_name=None):
        if type(label_list) is not list:
            raise ValueError(f"Argument `label_list` must be of type list. Got: f{type(label_list)}")

        if label_name is None:
            label_name = f"{name}_label"
        label_tensor_name = label_name + "_ids"
        self.tasks[name] = {
            "label_list": label_list,
            "metric": metric,
            "label_tensor_name": label_tensor_name,
            "label_name": label_name,
            "label_column_name": label_column_name,
            "text_column_name": text_column_name,
            "task_type": task_type,
        }

    @staticmethod
    def _check_sample_features(basket):
        """Check if all samples in the basket has computed its features.

        Args:
            basket: the basket containing the samples

        Returns:
            True if all the samples in the basket has computed its features, False otherwise

        """
        if len(basket.samples) == 0:
            return False
        for sample in basket.samples:
            if sample.features is None:
                return False
        return True

    @staticmethod
    def log_problematic(problematic_sample_ids):
        if problematic_sample_ids:
            n_problematic = len(problematic_sample_ids)
            problematic_id_str = ", ".join(problematic_sample_ids)
            logger.error(f"Unable to convert {n_problematic} samples to features. Their ids are : {problematic_id_str}")

    def _log_samples(self, n_samples):
        logger.info("*** Show {} random examples ***".format(n_samples))
        for i in range(n_samples):
            random_basket = random.choice(self.baskets)
            random_sample = random.choice(random_basket.samples)
            logger.info(random_sample)

    def _log_params(self):
        params = {
            "processor": self.__class__.__name__,
            "tokenizer": self.tokenizer.__class__.__name__,
        }
        names = ["max_seq_len", "dev_split"]
        for name in names:
            value = getattr(self, name)
            params.update({name: str(value)})
        if self.logger:
            self.logger.log_params(params)
