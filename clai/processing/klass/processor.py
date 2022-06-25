import logging

from clai.processing import base
from clai.tooling import io, proc

logger = logging.getLogger(__name__)


class TextClassificationProcessor(base.Processor, calling_name="klass"):
    """
    Used to handle the text classification datasets that come in tabular format (CSV, TSV, etc.)
    """

    def preprocess(self, txt):
        return txt

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.csv",
        dev_filename=None,
        test_filename="test.csv",
        dev_split=0.1,
        delimiter=",",
        quote_char="'",
        skiprows=None,
        label_column_name="label",
        multilabel=False,
        header=0,
        proxies=None,
        max_samples=None,
        text_column_name="text",
        **kwargs,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automaticaly
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict at
                         `farm.data_handler.utils.DOWNSTREAM_TASK_MAP <https://github.com/deepset-ai/FARM/blob/master/farm/data_handler/utils.py>`_.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file
        :type delimiter: str
        :param quote_char: Character used for quoting strings in the input tsv/ csv file
        :type quote_char: str
        :param skiprows: number of rows to skip in the tsvs (e.g. for multirow headers)
        :type skiprows: int
        :param label_column_name: name of the column in the input csv/tsv that shall be used as training labels
        :type label_column_name: str
        :param multilabel: set to True for multilabel classification
        :type multilabel: bool
        :param header: which line to use as a header in the input csv/tsv
        :type  header: int
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param text_column_name: name of the column in the input csv/tsv that shall be used as training text
        :type text_column_name: str
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        # TODO If an arg is misspelt, e.g. metrics, it will be swallowed silently by kwargs

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header
        self.max_samples = max_samples

        super(TextClassificationProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric and label_list:
            if multilabel:
                task_type = "multilabel_classification"
            else:
                task_type = "classification"
            self.add_task(
                name="text_classification",
                metric=metric,
                label_list=label_list,
                label_column_name=label_column_name,
                text_column_name=text_column_name,
                task_type=task_type,
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {}
        for task in self.tasks.values():
            column_mapping[task["label_column_name"]] = task["label_name"]
            column_mapping[task["text_column_name"]] = "text"
        data_dir = file.parent
        fname = file.stem
        suffix = file.suffix
        dicts = next(
            io.load(
                data_dir,
                fname,
                ext=suffix,
                quotechar='"',
                as_record=True,
                rename_columns=column_mapping,
                dtype=str,
                header=0,
            )
        )

        return dicts

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False, debug=False):
        self.baskets = []
        # Tokenize in batches
        texts = [self.preprocess(x["text"]) for x in dicts]
        tokenized_batch = self.tokenizer.batch_encode_plus(
            texts,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
        )
        input_ids_batch = tokenized_batch["input_ids"]
        segment_ids_batch = tokenized_batch["token_type_ids"]
        padding_masks_batch = tokenized_batch["attention_mask"]
        tokens_batch = [x.tokens for x in tokenized_batch.encodings]

        # From here we operate on a per sample basis
        for dictionary, input_ids, segment_ids, padding_mask, tokens in zip(
            dicts, input_ids_batch, segment_ids_batch, padding_masks_batch, tokens_batch
        ):

            tokenized = {}
            if debug:
                tokenized["tokens"] = tokens

            feat_dict = {"input_ids": input_ids, "padding_mask": padding_mask, "segment_ids": segment_ids}

            # Create labels
            # i.e. not inference
            if not return_baskets:
                label_dict = self.convert_labels(dictionary)
                feat_dict.update(label_dict)

            # Add Basket to self.baskets
            curr_sample = proc.Sample(id=None, clear_text=dictionary, tokenized=tokenized, features=[feat_dict])
            curr_basket = proc.SampleBasket(id_internal=None, raw=dictionary, id_external=None, samples=[curr_sample])
            self.baskets.append(curr_basket)

        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(1)

        # TODO populate problematic ids
        problematic_ids = set()
        logger.warning("Currently no support in Processor for returning problematic ids")
        dataset, tensornames = self._create_dataset()
        if return_baskets:
            return dataset, tensornames, problematic_ids, self.baskets
        else:
            return dataset, tensornames, problematic_ids

    def convert_labels(self, dictionary):
        ret = {}
        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            label_name = task["label_name"]
            label_raw = dictionary[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                label_ids = [label_list.index(label_raw)]
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            ret[task["label_tensor_name"]] = label_ids
        return ret

    def _create_dataset(self):
        # TODO this is the proposed new version to replace the mother function
        features_flat = []
        basket_to_remove = []
        for basket in self.baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        dataset, tensor_names = proc.convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names
