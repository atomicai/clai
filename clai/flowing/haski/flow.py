import copy
import logging
import pathlib

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from clai.flowing import flow
from clai.tooling import pic

logger = logging.getLogger(__name__)


class HaskiFlow(flow.Flow, signature="haski"):
    def __init__(
        self,
        processor,
        batch_size=1,
        eval_batch_size=None,
        distributed=False,
        automatic_loading=True,
        max_mp_size=2000,
        max_processes=128,
        caching=False,
        cache_path=pathlib.Path("cache/data_silo"),
        ml_logger=None,
    ):
        super().__init__(processor, max_processes=max_processes, max_mp_size=max_mp_size, ml_logger=ml_logger)
        self.distributed = distributed
        self.processor = processor
        self.data = {}
        self.batch_size = batch_size
        self.class_weights = None
        self.max_processes = max_processes
        self.max_multiprocessing_chunksize = max_mp_size
        self.caching = caching
        self.cache_path = cache_path
        self.tensor_names = None
        if eval_batch_size is None:
            self.eval_batch_size = batch_size
        else:
            self.eval_batch_size = eval_batch_size
        if len(self.processor.tasks) == 0:
            raise Exception(
                "No task initialized. Try initializing the processor with a metric and a label list. "
                "Alternatively you can add a task using Processor.add_task()"
            )

        loaded_from_cache = False
        if self.caching:  # Check if DataSets are present in cache
            checksum = self._get_checksum()
            dataset_path = self.cache_path / checksum

            if dataset_path.exists():
                self._load_dataset_from_cache(dataset_path)
                self.loaded_from_cache = True

        if not loaded_from_cache and automatic_loading:
            self._load_data()

    @classmethod
    def load(cls, processor, **kwargs):
        flow = cls(processor=processor, **kwargs)
        # TODO: add task
        return flow

    def _calc_length_stats_single_encoder(self):
        seq_lens = []
        for dataset in self.data["train"].datasets:
            train_input_numpy = dataset[:][self.tensor_names.index("input_ids")].numpy()
            seq_lens.extend(np.sum(train_input_numpy != self.processor.tokenizer.pad_token_id, axis=1))
        max_seq_len = dataset[:][self.tensor_names.index("input_ids")].shape[1]
        clipped = np.mean(np.array(seq_lens) == max_seq_len) if seq_lens else 0
        ave_len = np.mean(seq_lens) if seq_lens else 0
        return clipped, ave_len, seq_lens, max_seq_len

    def _calc_length_stats_biencoder(self):
        seq_lens = [[], []]
        for dataset in self.data["train"].datasets:
            query_input_numpy = dataset[:][self.tensor_names.index("query_input_ids")].numpy()
            num_passages = dataset[:][self.tensor_names.index("passage_input_ids")].shape[1]
            bs = dataset[:][self.tensor_names.index("passage_input_ids")].shape[0]
            passage_input_numpy = dataset[:][self.tensor_names.index("passage_input_ids")].numpy().reshape((bs, -1), order='C')
            qlen = np.sum(query_input_numpy != self.processor.query_tokenizer.pad_token_id, axis=1)
            plen = np.sum(passage_input_numpy != self.processor.passage_tokenizer.pad_token_id, axis=1) / num_passages
            seq_lens[0].extend(qlen)
            seq_lens[1].extend(plen)
        q_max_seq_len = dataset[:][self.tensor_names.index("query_input_ids")].shape[1]
        p_max_seq_len = dataset[:][self.tensor_names.index("passage_input_ids")].shape[2]
        clipped_q = np.mean(np.array(seq_lens[0]) == q_max_seq_len) if seq_lens[0] else 0
        ave_len_q = np.mean(seq_lens[0]) if seq_lens[0] else 0
        clipped_p = np.mean(np.array(seq_lens[1]) == p_max_seq_len) if seq_lens[1] else 0
        ave_len_p = np.mean(seq_lens[1]) if seq_lens[1] else 0
        clipped = [clipped_q, clipped_p]
        ave_len = [ave_len_q, ave_len_p]
        max_seq_len = [q_max_seq_len, p_max_seq_len]
        return clipped, ave_len, seq_lens, max_seq_len

    def calculate_class_weights(self, task_name, source="train"):
        """For imbalanced datasets, we can calculate class weights that can be used later in the
        loss function of the prediction head to upweight the loss of minorities.

        :param task_name: name of the task as used in the processor
        :type task_name: str
        """

        tensor_name = self.processor.tasks[task_name]["label_tensor_name"]
        label_list = self.processor.tasks[task_name]["label_list"]
        tensor_idx = list(self.tensor_names).index(tensor_name)
        # we need at least ONE observation for each label to avoid division by zero in compute_class_weights.
        observed_labels = copy.deepcopy(label_list)
        if source == "all":
            datasets = self.data.values()
        elif source == "train":
            datasets = [self.data["train"]]
        else:
            raise Exception("source argument expects one of [\"train\", \"all\"]")
        for dataset in datasets:
            observed_labels += [label_list[x[tensor_idx].item()] for x in dataset]

        # TODO scale e.g. via logarithm to avoid crazy spikes for rare classes
        class_weights = compute_class_weight("balanced", classes=np.asarray(label_list), y=observed_labels)

        # conversion necessary to have class weights of same type as model weights
        class_weights = class_weights.astype(np.float32)
        return class_weights

    def get_data_loader(self, dataset_name: str, **kwargs):
        return self.loaders[dataset_name]

    def _load_data(self, train_dicts=None, dev_dicts=None, test_dicts=None):
        """
        Loading the train, dev and test datasets either from files (default) or from supplied dicts.
        The processor is called to handle the full conversion from "raw data" to a Pytorch Dataset.
        The resulting datasets are loaded into DataSilo.data

        :param train_dicts: (Optional) dicts containing examples for training.
        :param dev_dicts: (Optional) dicts containing examples for dev.
        :param test_dicts: (Optional) dicts containing examples for test.
        :return: None
        """

        logger.info("\nLoading data into the data silo ..." "{}".format(pic.TRACTOR_SMALL))
        # train data
        logger.info("LOADING TRAIN DATA")
        logger.info("==================")
        if train_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["train"], self.tensor_names = self._get_dataset(filename=None, dicts=train_dicts)
        elif self.processor.train_filename:
            # or from a file (default)
            train_file = self.processor.data_dir / self.processor.train_filename
            logger.info("Loading train set from: {} ".format(train_file))
            self.data["train"], self.tensor_names = self._get_dataset(train_file)
        else:
            logger.info("No train set is being loaded")
            self.data["train"] = None

        # dev data
        logger.info("")
        logger.info("LOADING DEV DATA")
        logger.info("=================")
        if dev_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["dev"], self.tensor_names = self._get_dataset(filename=None, dicts=dev_dicts)
        elif self.processor.dev_filename:
            # or from file (default)
            dev_file = self.processor.data_dir / self.processor.dev_filename
            logger.info("Loading dev set from: {}".format(dev_file))
            self.data["dev"], _ = self._get_dataset(dev_file)
        elif self.processor.dev_split > 0.0:
            # or split it apart from train set
            logger.info("Loading dev set as a slice of train set")
            self._create_dev_from_train()
        else:
            logger.info("No dev set is being loaded")
            self.data["dev"] = None

        logger.info("")
        logger.info("LOADING TEST DATA")
        logger.info("=================")
        # test data
        if test_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["test"], self.tensor_names = self._get_dataset(filename=None, dicts=test_dicts)
        elif self.processor.test_filename:
            # or from file (default)
            test_file = self.processor.data_dir / self.processor.test_filename
            logger.info("Loading test set from: {}".format(test_file))
            if self.tensor_names:
                self.data["test"], _ = self._get_dataset(test_file)
            else:
                self.data["test"], self.tensor_names = self._get_dataset(test_file)
        else:
            logger.info("No test set is being loaded")
            self.data["test"] = None

        if self.caching:
            self._save_dataset_to_cache()

        # derive stats and meta data
        self._calculate_statistics()
        # self.calculate_class_weights()

        self._initialize_data_loaders()

    def n_samples(self, dataset_name):
        """
        Returns the number of samples in a given dataset.

        :param dataset_name: Choose from train, dev or test
        :type dataset_name: str
        """
        return self.counts[dataset_name]
