import logging
import multiprocessing as mp
import numbers

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from farm.utils import flatten_list

logger = logging.getLogger(__name__)

SAMPLE = """
      .--.        _____                       _      
    .'_\/_'.     / ____|                     | |     
    '. /\ .'    | (___   __ _ _ __ ___  _ __ | | ___ 
      "||"       \___ \ / _` | '_ ` _ \| '_ \| |/ _ \ 
       || /\     ____) | (_| | | | | | | |_) | |  __/
    /\ ||//\)   |_____/ \__,_|_| |_| |_| .__/|_|\___|
   (/\\||/                             |_|           
______\||/___________________________________________                     
"""


class SampleBasket:
    """An object that contains one source text and the one or more samples that will be processed. This
    is needed for tasks like question answering where the source text can generate multiple input - label
    pairs."""

    def __init__(self, id_internal: str, raw: dict, id_external=None, samples=None):
        """
        :param id_internal: A unique identifying id. Used for identification within FARM.
        :type id_internal: str
        :param external_id: Used for identification outside of FARM. E.g. if another framework wants to pass along its own id with the results.
        :type external_id: str
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :type raw: dict
        :param samples: An optional list of Samples used to populate the basket at initialization.
        :type samples: Sample
        """
        self.id_internal = id_internal
        self.id_external = id_external
        self.raw = raw
        self.samples = samples


class Sample(object):
    """A single training/test sample. This should contain the input and the label. Is initialized with
    the human readable clear_text. Over the course of data preprocessing, this object is populated
    with tokenized and featurized versions of the data."""

    def __init__(self, id, clear_text, tokenized=None, features=None):
        """
        :param id: The unique id of the sample
        :type id: str
        :param clear_text: A dictionary containing various human readable fields (e.g. text, label).
        :type clear_text: dict
        :param tokenized: A dictionary containing the tokenized version of clear text plus helpful meta data: offsets (start position of each token in the original text) and start_of_word (boolean if a token is the first one of a word).
        :type tokenized: dict
        :param features: A dictionary containing features in a vectorized format needed by the model to process this sample.
        :type features: dict

        """
        self.id = id
        self.clear_text = clear_text
        self.features = features
        self.tokenized = tokenized

    def __str__(self):

        if self.clear_text:
            clear_text_str = "\n \t".join([k + ": " + str(v) for k, v in self.clear_text.items()])
            if len(clear_text_str) > 10000:
                clear_text_str = (
                    clear_text_str[:10_000] + f"\nTHE REST IS TOO LONG TO DISPLAY. "
                    f"Remaining chars :{len(clear_text_str)-10_000}"
                )
        else:
            clear_text_str = "None"

        if self.features:
            if isinstance(self.features, list):
                features = self.features[0]
            else:
                features = self.features
            feature_str = "\n \t".join([k + ": " + str(v) for k, v in features.items()])
        else:
            feature_str = "None"

        if self.tokenized:
            tokenized_str = "\n \t".join([k + ": " + str(v) for k, v in self.tokenized.items()])
            if len(tokenized_str) > 10000:
                tokenized_str = (
                    tokenized_str[:10_000] + f"\nTHE REST IS TOO LONG TO DISPLAY. "
                    f"Remaining chars: {len(tokenized_str)-10_000}"
                )
        else:
            tokenized_str = "None"
        s = (
            f"\n{SAMPLE}\n"
            f"ID: {self.id}\n"
            f"Clear Text: \n \t{clear_text_str}\n"
            f"Tokenized: \n \t{tokenized_str}\n"
            f"Features: \n \t{feature_str}\n"
            "_____________________________________________________"
        )
        return s


def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs (e.g. Natural Questions downsamples instances of is_impossible)
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        # Conversion of floats
        if t_name == "regression_label_ids":
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.float32)
        else:
            try:
                # Checking weather a non-integer will be silently converted to torch.long
                check = features[0][t_name]
                if isinstance(check, numbers.Number):
                    base = check
                # extract a base variable from a nested lists or tuples
                elif isinstance(check, list):
                    base = list(flatten_list(check))[0]
                # extract a base variable from numpy arrays
                else:
                    base = check.ravel()[0]
                if not np.issubdtype(type(base), np.integer):
                    logger.warning(
                        f"Problem during conversion to torch tensors:\n"
                        f"A non-integer value for feature '{t_name}' with a value of: "
                        f"'{base}' will be converted to a torch tensor of dtype long."
                    )
            except:
                logger.warning(
                    f"Could not determine type for feature '{t_name}'. Converting now to a tensor of default type long."
                )

            # Convert all remaining python objects to torch long tensors
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.long)

        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


class NamedDataLoader(DataLoader):
    """
    A modified version of the PyTorch DataLoader that returns a dictionary where the key is
    the name of the tensor and the value is the tensor itself.
    """

    def __init__(self, dataset, batch_size, sampler=None, tensor_names=None, num_workers=0, pin_memory=False):
        """
        :param dataset: The dataset that will be wrapped by this NamedDataLoader
        :type dataset: Dataset
        :param sampler: The sampler used by the NamedDataLoader to choose which samples to include in the batch
        :type sampler: Sampler
        :param batch_size: The size of the batch to be returned by the NamedDataLoader
        :type batch_size: int
        :param tensor_names: The names of the tensor, in the order that the dataset returns them in.
        :type tensor_names: list
        :param num_workers: number of workers to use for the DataLoader
        :type num_workers: int
        :param pin_memory: argument for Data Loader to use page-locked memory for faster transfer of data to GPU
        :type pin_memory: bool
        """

        def collate_fn(batch):
            """
            A custom collate function that formats the batch as a dictionary where the key is
            the name of the tensor and the value is the tensor itself
            """

            if type(dataset).__name__ == "_StreamingDataSet":
                _tensor_names = dataset.tensor_names
            else:
                _tensor_names = tensor_names

            if type(batch[0]) == list:
                batch = batch[0]

            assert len(batch[0]) == len(
                _tensor_names
            ), "Dataset contains {} tensors while there are {} tensor names supplied: {}".format(
                len(batch[0]), len(_tensor_names), _tensor_names
            )
            lists_temp = [[] for _ in range(len(_tensor_names))]
            ret = dict(zip(_tensor_names, lists_temp))

            for example in batch:
                for name, tensor in zip(_tensor_names, example):
                    ret[name].append(tensor)

            for key in ret:
                ret[key] = torch.stack(ret[key])

            return ret

        super(NamedDataLoader, self).__init__(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def __len__(self):
        if type(self.dataset).__name__ == "_StreamingDataSet":
            num_samples = len(self.dataset)
            num_batches = np.ceil(num_samples / self.dataset.batch_size)
            return num_batches
        else:
            return super().__len__()


def calc_chunksize(num_dicts, min_chunksize=4, max_chunksize=2000, max_processes=128):
    if mp.cpu_count() > 3:
        num_cpus = min(mp.cpu_count() - 1 or 1, max_processes)  # -1 to keep a CPU core free for xxx
    else:
        num_cpus = min(mp.cpu_count(), max_processes)  # when there are few cores, we use all of them

    dicts_per_cpu = np.ceil(num_dicts / num_cpus)
    # automatic adjustment of multiprocessing chunksize
    # for small files (containing few dicts) we want small chunksize to ulitize all available cores but never less
    # than 2, because we need it to sample another random sentence in LM finetuning
    # for large files we want to minimize processor spawning without giving too much data to one process, so we
    # clip it at 5k
    multiprocessing_chunk_size = int(np.clip((np.ceil(dicts_per_cpu / 5)), a_min=min_chunksize, a_max=max_chunksize))
    # This lets us avoid cases in lm_finetuning where a chunk only has a single doc and hence cannot pick
    # a valid next sentence substitute from another document
    if num_dicts != 1:
        while num_dicts % multiprocessing_chunk_size == 1:
            multiprocessing_chunk_size -= -1
    dict_batches_to_process = int(num_dicts / multiprocessing_chunk_size)
    num_processes = min(num_cpus, dict_batches_to_process) or 1

    return multiprocessing_chunk_size, num_processes


def covert_dataset_to_dataloader(dataset, sampler, batch_size):
    """
    Wraps a PyTorch Dataset with a DataLoader.

    :param dataset: Dataset to be wrapped.
    :type dataset: Dataset
    :param sampler: PyTorch sampler used to pick samples in a batch.
    :type sampler: Sampler
    :param batch_size: Number of samples in the batch.
    :return: A DataLoader that wraps the input Dataset.
    """
    sampler_initialized = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler_initialized, batch_size=batch_size)
    return data_loader


def create_sample_one_label_one_text(raw_data, text_index, label_index, basket_id):

    # text = " ".join(raw_data[text_index:])
    text = raw_data[text_index]
    label = raw_data[label_index]

    return [Sample(id=basket_id + "-0", clear_text={"text": text, "label": label})]


def create_sample_ner(split_text, label, basket_id):

    text = " ".join(split_text)
    label = label

    return [Sample(id=basket_id + "-0", clear_text={"text": text, "label": label})]


def process_answers(answers, doc_offsets, passage_start_c, passage_start_t):
    """TODO Write Comment"""
    answers_clear = []
    answers_tokenized = []
    for answer in answers:
        # This section calculates start and end relative to document
        answer_text = answer["text"]
        answer_len_c = len(answer_text)
        if "offset" in answer:
            answer_start_c = answer["offset"]
        else:
            answer_start_c = answer["answer_start"]
        answer_end_c = answer_start_c + answer_len_c - 1
        answer_start_t = offset_to_token_idx_vecorized(doc_offsets, answer_start_c)
        answer_end_t = offset_to_token_idx_vecorized(doc_offsets, answer_end_c)

        # # Leaving this code for potentially debugging 'offset_to_token_idx_vecorized()'
        # answer_start_t2 = offset_to_token_idx(doc_offsets, answer_start_c)
        # answer_end_t2 = offset_to_token_idx(doc_offsets, answer_end_c)
        # if (answer_start_t != answer_start_t2) or (answer_end_t != answer_end_t2):
        #     pass

        # TODO: Perform check that answer can be recovered from document?
        # This section converts start and end so that they are relative to the passage
        # TODO: Is this actually necessary on character level?
        answer_start_c -= passage_start_c
        answer_end_c -= passage_start_c
        answer_start_t -= passage_start_t
        answer_end_t -= passage_start_t

        curr_answer_clear = {"text": answer_text, "start_c": answer_start_c, "end_c": answer_end_c}
        curr_answer_tokenized = {
            "start_t": answer_start_t,
            "end_t": answer_end_t,
            "answer_type": answer.get("answer_type", "span"),
        }

        answers_clear.append(curr_answer_clear)
        answers_tokenized.append(curr_answer_tokenized)
    return answers_clear, answers_tokenized


def get_passage_offsets(doc_offsets, doc_stride, passage_len_t, doc_text):
    """
    Get spans (start and end offsets) for passages by applying a sliding window function.
    The sliding window moves in steps of doc_stride.
    Returns a list of dictionaries which each describe the start, end and id of a passage
    that is formed when chunking a document using a sliding window approach."""

    passage_spans = []
    passage_id = 0
    doc_len_t = len(doc_offsets)
    while True:
        passage_start_t = passage_id * doc_stride
        passage_end_t = passage_start_t + passage_len_t
        passage_start_c = doc_offsets[passage_start_t]

        # If passage_end_t points to the last token in the passage, define passage_end_c as the length of the document
        if passage_end_t >= doc_len_t - 1:
            passage_end_c = len(doc_text)

        # Get document text up to the first token that is outside the passage. Strip of whitespace.
        # Use the length of this text as the passage_end_c
        else:
            end_ch_idx = doc_offsets[passage_end_t + 1]
            raw_passage_text = doc_text[:end_ch_idx]
            passage_end_c = len(raw_passage_text.strip())

        passage_span = {
            "passage_start_t": passage_start_t,
            "passage_end_t": passage_end_t,
            "passage_start_c": passage_start_c,
            "passage_end_c": passage_end_c,
            "passage_id": passage_id,
        }
        passage_spans.append(passage_span)
        passage_id += 1
        # If the end idx is greater than or equal to the length of the passage
        if passage_end_t >= doc_len_t:
            break
    return passage_spans


def offset_to_token_idx(token_offsets, ch_idx):
    """Returns the idx of the token at the given character idx"""
    n_tokens = len(token_offsets)
    for i in range(n_tokens):
        if (i + 1 == n_tokens) or (token_offsets[i] <= ch_idx < token_offsets[i + 1]):
            return i


def offset_to_token_idx_vecorized(token_offsets, ch_idx):
    """Returns the idx of the token at the given character idx"""
    ################
    ################
    ##################
    # TODO CHECK THIS fct thoroughly - This must be bulletproof and inlcude start and end of sequence checks
    # todo Possibly this function does not work for Natural Questions and needs adjustments
    ################
    ################
    ##################
    # case ch_idx is at end of tokens
    if ch_idx >= np.max(token_offsets):
        # TODO check "+ 1" (it is needed for making end indices compliant with old offset_to_token_idx() function)
        # check weather end token is incluse or exclusive
        idx = np.argmax(token_offsets) + 1
    # looking for the first occurence of token_offsets larger than ch_idx and taking one position to the left.
    # This is needed to overcome n special_tokens at start of sequence
    # and failsafe matching (the character start might not always coincide with a token offset, e.g. when starting at whitespace)
    else:
        idx = np.argmax(token_offsets > ch_idx) - 1
    return idx


def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs (e.g. Natural Questions downsamples instances of is_impossible)
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        # Conversion of floats
        if t_name == "regression_label_ids":
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.float32)
        else:
            try:
                # Checking weather a non-integer will be silently converted to torch.long
                check = features[0][t_name]
                if isinstance(check, numbers.Number):
                    base = check
                # extract a base variable from a nested lists or tuples
                elif isinstance(check, list):
                    base = list(flatten_list(check))[0]
                # extract a base variable from numpy arrays
                else:
                    base = check.ravel()[0]
                if not np.issubdtype(type(base), np.integer):
                    logger.warning(
                        f"Problem during conversion to torch tensors:\n"
                        f"A non-integer value for feature '{t_name}' with a value of: "
                        f"'{base}' will be converted to a torch tensor of dtype long."
                    )
            except:
                logger.warning(
                    f"Could not determine type for feature '{t_name}'. Converting now to a tensor of default type long."
                )

            # Convert all remaining python objects to torch long tensors
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.long)

        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names
