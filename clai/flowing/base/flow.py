import abc
import logging
import multiprocessing as mp
import pathlib
import random
from contextlib import ExitStack
from functools import partial

from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.autonotebook import tqdm

from clai.tooling import proc, tool

logger = logging.getLogger(__name__)

# TODO:
# Right now the base flow is both `haski` and `stream` at the same time.
# With that in mind it is, therefore, necessary to move the `dataset_from_chunk(...)`
# to `haski` instance while other single process implementation to `stream`


class Flow(abc.ABC):

    subclasses = {}

    def __init_subclass__(cls, signature: str = None, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific `Flow` implementation.
        """
        super().__init_subclass__(**kwargs)
        signature = cls.__name__ if signature is None else signature
        cls.subclasses[signature] = cls

    @classmethod
    def load(cls, name: str = None, **kwargs):
        klass = cls.subclasses[name] if name is not None else cls.subclasses[cls.__name__]
        return klass.load(**kwargs)

    def __init__(self, processor, max_processes, max_mp_size, ml_logger=None):
        self.processor = processor
        self.max_processes = max_processes
        self.max_mp_size = max_mp_size
        self.logger = ml_logger

    def _calculate_statistics(self):
        """Calculate and log simple summary statistics of the datasets"""
        logger.info("")
        logger.info("DATASETS SUMMARY")
        logger.info("================")

        self.counts = {}
        clipped = -1
        ave_len = -1

        if self.data["train"]:
            self.counts["train"] = len(self.data["train"])
            if "input_ids" in self.tensor_names:
                clipped, ave_len, seq_lens, max_seq_len = self._calc_length_stats_single_encoder()
            elif "query_input_ids" in self.tensor_names and "passage_input_ids" in self.tensor_names:
                clipped, ave_len, seq_lens, max_seq_len = self._calc_length_stats_biencoder()
            else:
                logger.warning(
                    f"Could not compute length statistics because 'input_ids' or 'query_input_ids' and 'passage_input_ids' are missing."
                )
                clipped = -1
                ave_len = -1
        else:
            self.counts["train"] = 0

        if self.data["dev"]:
            self.counts["dev"] = len(self.data["dev"])
        else:
            self.counts["dev"] = 0

        if self.data["test"]:
            self.counts["test"] = len(self.data["test"])
        else:
            self.counts["test"] = 0

        logger.info("Examples in train: {}".format(self.counts["train"]))
        logger.info("Examples in dev  : {}".format(self.counts["dev"]))
        logger.info("Examples in test : {}".format(self.counts["test"]))
        logger.info("")
        if self.data["train"]:
            if "input_ids" in self.tensor_names:
                logger.info("Longest sequence length observed after clipping:     {}".format(max(seq_lens)))
                logger.info("Average sequence length after clipping: {}".format(ave_len))
                logger.info("Proportion clipped:      {}".format(clipped))
                if clipped > 0.5:
                    logger.info(
                        "[Farmer's Tip] {}% of your samples got cut down to {} tokens. "
                        "Consider increasing max_seq_len. "
                        "This will lead to higher memory consumption but is likely to "
                        "improve your model performance".format(round(clipped * 100, 1), max_seq_len)
                    )
            elif "query_input_ids" in self.tensor_names and "passage_input_ids" in self.tensor_names:
                logger.info(
                    "Longest query length observed after clipping: {}   - for max_query_len: {}".format(
                        max(seq_lens[0]), max_seq_len[0]
                    )
                )
                logger.info("Average query length after clipping:          {}".format(ave_len[0]))
                logger.info("Proportion queries clipped:                   {}".format(clipped[0]))
                logger.info("")
                logger.info(
                    "Longest passage length observed after clipping: {}   - for max_passage_len: {}".format(
                        max(seq_lens[1]), max_seq_len[1]
                    )
                )
                logger.info("Average passage length after clipping:          {}".format(ave_len[1]))
                logger.info("Proportion passages clipped:                    {}".format(clipped[1]))

        if self.logger:
            self.logger.log_params(
                {
                    "n_samples_train": self.counts["train"],
                    "n_samples_dev": self.counts["dev"],
                    "n_samples_test": self.counts["test"],
                    "batch_size": self.batch_size,
                    "ave_seq_len": ave_len,
                    "clipped": clipped,
                }
            )

    def _initialize_data_loaders(self):
        """Initializing train, dev and test data loaders for the already loaded datasets"""

        if self.data["train"] is not None:
            if self.distributed:
                sampler_train = DistributedSampler(self.data["train"])
            else:
                sampler_train = RandomSampler(self.data["train"])

            data_loader_train = proc.NamedDataLoader(
                dataset=self.data["train"],
                sampler=sampler_train,
                batch_size=self.batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_train = None

        if self.data["dev"] is not None:
            data_loader_dev = proc.NamedDataLoader(
                dataset=self.data["dev"],
                sampler=SequentialSampler(self.data["dev"]),
                batch_size=self.eval_batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_dev = None

        if self.data["test"] is not None:
            data_loader_test = proc.NamedDataLoader(
                dataset=self.data["test"],
                sampler=SequentialSampler(self.data["test"]),
                batch_size=self.eval_batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_test = None

        self.loaders = {
            "train": data_loader_train,
            "dev": data_loader_dev,
            "test": data_loader_test,
        }

    def _get_dataset(self, filename, dicts=None):
        if not filename and not dicts:
            raise ValueError("You must either supply `filename` or `dicts`")

        # loading dicts from file (default)
        if dicts is None:
            dicts = list(self.processor.file_to_dicts(filename))
            # shuffle list of dicts here if we later want to have a random dev set splitted from train set
            if str(self.processor.train_filename) in str(filename):
                if not self.processor.dev_filename:
                    if self.processor.dev_split > 0.0:
                        random.shuffle(dicts)

        num_dicts = len(dicts)
        multiprocessing_chunk_size, num_cpus_used = proc.calc_chunksize(
            num_dicts=num_dicts,
            max_processes=self.max_processes,
            max_chunksize=self.max_mp_size,
        )

        with ExitStack() as stack:
            if self.max_processes > 1:  # use multiprocessing only when max_processes > 1
                p = stack.enter_context(mp.Pool(processes=num_cpus_used))

                logger.info(
                    f"Got ya {num_cpus_used} parallel workers to convert {num_dicts} dictionaries "
                    f"to pytorch datasets (chunksize = {multiprocessing_chunk_size})..."
                )

                results = p.imap(
                    partial(self._dataset_from_chunk, processor=self.processor),
                    tool.grouper(dicts, multiprocessing_chunk_size),
                    chunksize=1,
                )
            else:
                logger.info(
                    f"Multiprocessing disabled, using a single worker to convert {num_dicts}"
                    f" dictionaries to pytorch datasets."
                )

                results = map(partial(self._dataset_from_chunk, processor=self.processor), tool.grouper(dicts, num_dicts))

            datasets = []
            problematic_ids_all = set()

            desc = f"Preprocessing Dataset"
            if filename:
                desc += f" {filename}"
            with tqdm(total=len(dicts), unit=' Dicts', desc=desc) as pbar:
                for dataset, tensor_names, problematic_samples in results:
                    datasets.append(dataset)
                    # update progress bar (last step can have less dicts than actual chunk_size)
                    pbar.update(min(multiprocessing_chunk_size, pbar.total - pbar.n))
                    problematic_ids_all.update(problematic_samples)

            self.processor.log_problematic(problematic_ids_all)
            # _dataset_from_chunk can return a None in cases where downsampling has occurred
            datasets = [d for d in datasets if d]
            concat_datasets = ConcatDataset(datasets)
            return concat_datasets, tensor_names

    @classmethod
    def _dataset_from_chunk(cls, chunk, processor):
        """
        Creating a dataset for a chunk (= subset) of dicts. In multiprocessing:
          * we read in all dicts from a file
          * split all dicts into chunks
          * feed *one chunk* to *one process*
          => the *one chunk*  gets converted to *one dataset* (that's what we do here)
          * all datasets get collected and concatenated
        :param chunk: Instead of only having a list of dicts here we also supply an index (ascending int) for each.
            => [(0, dict), (1, dict) ...]
        :type chunk: list of tuples
        :param processor: FARM Processor (e.g. TextClassificationProcessor)
        :return: PyTorch Dataset
        """
        dicts = [d[1] for d in chunk]
        indices = [x[0] for x in chunk]
        dataset, tensor_names, problematic_sample_ids = processor.dataset_from_dicts(dicts=dicts, indices=indices)
        return dataset, tensor_names, problematic_sample_ids

    def _get_checksum(self):
        """
        Get checksum based on a dict to ensure validity of cached `flow`
        """
        # keys in the dict identifies uniqueness for a given `Flow`.
        payload_dict = {
            "train_filename": str(pathlib.Path(self.processor.train_filename).absolute()),
            "data_dir": str(self.processor.data_dir.absolute()),
            "max_seq_len": self.processor.max_seq_len,
            "dev_split": self.processor.dev_split,
            "tasks": self.processor.tasks,
        }
        checksum = tool.get_dict_checksum(payload_dict)
        return checksum

    @abc.abstractmethod
    def calculate_class_weights(self, task_name, source="train"):
        pass

    @abc.abstractmethod
    def get_data_loader(self, dataset_name: str):
        pass
