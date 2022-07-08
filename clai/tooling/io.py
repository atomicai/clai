import ast
import copy
import json
import logging
import pathlib
from typing import Union

import numpy as np
import pandas as pd
import random_name
import simplejson
from clai.tooling import pic, remote
from transformers import (
    AlbertTokenizer,
    AlbertTokenizerFast,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    BigBirdTokenizer,
    BigBirdTokenizerFast,
    CamembertTokenizer,
    CamembertTokenizerFast,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    ElectraTokenizer,
    ElectraTokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizerFast,
    XLMTokenizer,
    XLNetTokenizer,
    XLNetTokenizerFast,
)

logger = logging.getLogger(__name__)


def load_tokenizer(pretrained_model_name_or_path, revision=None, tokenizer_class=None, use_fast=True, **kwargs):
    """
    Enables loading of different Tokenizer classes with a uniform interface. Either infer the class from
    model config or define it manually via `tokenizer_class`.

    :param pretrained_model_name_or_path:  The path of the saved pretrained model or its name (e.g. `bert-base-uncased`)
    :type pretrained_model_name_or_path: str
    :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
    :type revision: str
    :param tokenizer_class: (Optional) Name of the tokenizer class to load (e.g. `BertTokenizer`)
    :type tokenizer_class: str
    :param use_fast: (Optional, False by default) Indicate if FARM should try to load the fast version of the tokenizer (True) or
        use the Python one (False).
        Only DistilBERT, BERT and Electra fast tokenizers are supported.
    :type use_fast: bool
    :param kwargs:
    :return: Tokenizer
    """

    def _get_tok(pretrained_model_name_or_path):
        """
        Infer the tokenizer type
        """
        config, tokenizer_class = None, None
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except OSError:
            # FARM model (no 'config.json' file)
            try:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path + "/language_model_config.json")
            except Exception:
                logger.warning("No config file found. Trying to infer Tokenizer type from model name")

        model_type = config.model_type
        if model_type == "xlm-roberta":
            tokenizer_class = "XLMRobertaTokenizer"
        elif model_type == "roberta":
            if "mlm" in pretrained_model_name_or_path.lower():
                raise NotImplementedError("MLM part of codebert is currently not supported in FARM")
            tokenizer_class = RobertaTokenizer if not use_fast else RobertaTokenizerFast
        elif model_type == "camembert":
            tokenizer_class = CamembertTokenizer if not use_fast else CamembertTokenizerFast
        elif model_type == "albert":
            tokenizer_class = AlbertTokenizer if not use_fast else AlbertTokenizerFast
        elif model_type == "distilbert":
            tokenizer_class = DistilBertTokenizer if not use_fast else DistilBertTokenizerFast
        elif model_type == "bert":
            tokenizer_class = BertTokenizer if not use_fast else BertTokenizerFast
        elif model_type == "xlnet":
            tokenizer_class = XLNetTokenizer if not use_fast else XLNetTokenizerFast
        elif model_type == "electra":
            tokenizer_class = ElectraTokenizer if not use_fast else ElectraTokenizerFast
        elif model_type == "big_bird":
            tokenizer_class = BigBirdTokenizer if not use_fast else BigBirdTokenizerFast
        else:
            raise ValueError(f"Couldn't infer the tokenizer class from model type `{model_type}`")
        return tokenizer_class

    # TODO: assert correct version of the tokenizer
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    kwargs["revision"] = revision

    klass = _get_tok(pretrained_model_name_or_path)
    return klass.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast)


def prepare_if_missing(where: Union[pathlib.Path, str]):
    path, status = pathlib.Path(where), False
    try:
        path.mkdir(parents=True, exist_ok=False)
        status = True
    except FileExistsError:
        status = False
    else:
        status = True
    return status


def load(
    data_dir: Union[pathlib.Path, str],
    filename: str,
    embedding_field="embedding",
    load_embedding=True,
    ext=".json",
    parse_meta: bool = False,
    lazy: bool = False,
    sep: str = ",",
    encoding: str = "utf-8-sig",
    as_record: bool = False,
    rename_columns: dict = None,
    **kwargs,
):
    data_dir = pathlib.Path(data_dir)
    db_filename = filename
    db_filepath = data_dir / (db_filename + ext)

    if ext in (".csv", ".tsv"):
        columns_needed = list(rename_columns.keys()) if rename_columns else None

        df = pd.read_csv(db_filepath, encoding=encoding, skipinitialspace=True, sep=sep, usecols=columns_needed, **kwargs)
        df = df.rename(columns=rename_columns) if rename_columns else df
        if as_record:
            yield df.to_dict(orient="records")
        else:
            yield df
        raise StopIteration()
    with open(str(db_filepath), "r", encoding=encoding) as j_ptr:
        if lazy:
            for jline in j_ptr:
                yield json.loads(jline)
        else:
            docs = json.load(j_ptr)

    if lazy:
        raise StopIteration

    if parse_meta:
        for d in docs:
            d["meta"] = ast.literal_eval(d["meta"])

    if embedding_field is not None:
        if load_embedding:
            index_filename = filename + "_index" + ".npy"
            index_filepath = data_dir / index_filename
            embeddings = np.load(str(index_filepath))
            for iDoc, iEmb in zip(docs, embeddings):
                iDoc[embedding_field] = iEmb
        else:
            for iDoc in docs:
                iDoc[embedding_field] = np.nan

    yield docs


def save(data, data_dir: Union[str, pathlib.Path], embedding_field="embedding", save_embedding=True, ext=".json"):
    data_dir = pathlib.Path(data_dir)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    db_filename = random_name.generate_name()

    db_filepath = data_dir / (db_filename + ext)

    if embedding_field is not None:
        if save_embedding:
            index_filename = db_filename + "_index" + ".npy"
            index_filepath = data_dir / index_filename
            index_data = []
            for dic in data:
                index_data.append(copy.deepcopy(dic[embedding_field]))
                dic[embedding_field] = np.nan
            np.save(index_filepath, np.array(index_data))
        else:
            for dic in data:
                dic[embedding_field] = np.nan
    else:
        pass

    with open(str(db_filepath), "w", encoding="utf-8") as j_ptr:
        simplejson.dump(data, j_ptr, indent=4, ensure_ascii=False, ignore_nan=True)


def chunkify(f, chunksize=10_000_000, sep="\n"):
    """
    Read a file separating its content lazily.

    Usage:

    >>> with open('INPUT.TXT') as f:
    >>>     for item in chunkify(f):
    >>>         process(item)
    """
    chunk = None
    remainder = None  # data from the previous chunk.
    while chunk != "":
        chunk = f.read(chunksize)
        if remainder:
            piece = remainder + chunk
        else:
            piece = chunk
        pos = None
        while pos is None or pos >= 0:
            pos = piece.find(sep)
            if pos >= 0:
                if pos > 0:
                    yield piece[:pos]
                piece = piece[pos + 1 :]
                remainder = None
            else:
                remainder = piece
    if remainder:  # This statement will be executed iff @remainder != ''
        yield remainder


def log_ascii_workers(n, logger):
    m_worker_lines = pic.WORKER_M.split("\n")
    f_worker_lines = pic.WORKER_F.split("\n")
    x_worker_lines = pic.WORKER_X.split("\n")
    all_worker_lines = []
    for i in range(n):
        rand = np.random.randint(low=0, high=3)
        if rand % 3 == 0:
            all_worker_lines.append(f_worker_lines)
        elif rand % 3 == 1:
            all_worker_lines.append(m_worker_lines)
        else:
            all_worker_lines.append(x_worker_lines)
    zipped = zip(*all_worker_lines)
    for z in zipped:
        logger.info("  ".join(z))


def format_log(ascii, logger):
    ascii_lines = ascii.split("\n")
    for l in ascii_lines:
        logger.info(l)
