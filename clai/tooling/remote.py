import hashlib
import logging
import os
import pathlib
import tarfile
import tempfile

import requests
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)

DOWNSTREAM_TASK_MAP = {
    "gnad": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/gnad.tar.gz",
    "germeval14": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval14.tar.gz",
    # only has train.tsv and test.tsv dataset - no dev.tsv
    "germeval18": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval18.tar.gz",
    "squad20": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz",
    "covidqa": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/covidqa.tar.gz",
    "conll03detrain": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.train",
    "conll03dedev": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testa",  # https://www.clips.uantwerpen.be/conll2003/ner/000README says testa is dev data
    "conll03detest": "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testb",
    "conll03entrain": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train",
    "conll03endev": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa",
    "conll03entest": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb",
    "cord_19": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cord_19.tar.gz",
    "lm_finetune_nips": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/lm_finetune_nips.tar.gz",
    "toxic-comments": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/toxic-comments.tar.gz",
    "cola": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cola.tar.gz",
    "asnq_binary": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/asnq_binary.tar.gz",
    "germeval17": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval17.tar.gz",
    "natural_questions": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/natural_questions.tar.gz",
}


def _conll03get(dataset, directory, language):
    # open in binary mode
    with open(directory / f"{dataset}.txt", "wb") as f:
        # get request
        response = requests.get(DOWNSTREAM_TASK_MAP[f"conll03{language}{dataset}"])
        # write to file
        f.write(response.content)

    # checking files for correctness with md5sum.
    if f"conll03{language}{dataset}" == "conll03detrain":
        if "ae4be68b11dc94e0001568a9095eb391" != _get_md5checksum(str(directory / f"{dataset}.txt")):
            logger.error(
                f"Someone has changed the file for conll03detrain. This data was collected from an external github repository.\n"
                f"Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py"
            )
    elif f"conll03{language}{dataset}" == "conll03detest":
        if "b8514f44366feae8f317e767cf425f28" != _get_md5checksum(str(directory / f"{dataset}.txt")):
            logger.error(
                f"Someone has changed the file for conll03detest. This data was collected from an external github repository.\n"
                f"Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py"
            )
    elif f"conll03{language}{dataset}" == "conll03entrain":
        if "11a942ce9db6cc64270372825e964d26" != _get_md5checksum(str(directory / f"{dataset}.txt")):
            logger.error(
                f"Someone has changed the file for conll03entrain. This data was collected from an external github repository.\n"
                f"Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py"
            )


def http_get(url, temp_file, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def _get_md5checksum(fname):
    # solution from stackoverflow: https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download_extract_downstream_data(input_file, proxies=None):
    # download archive to temp dir and extract to correct position
    full_path = pathlib.Path(os.path.realpath(input_file))
    directory = full_path.parent
    taskname = directory.stem
    datadir = directory.parent
    logger.info("downloading and extracting file {} to dir {}".format(taskname, datadir))
    if "conll03-" in taskname:
        # conll03 is copyrighted, but luckily somebody put it on github. Kudos!
        if not os.path.exists(directory):
            os.makedirs(directory)
        for dataset in ["train", "dev", "test"]:
            if "de" in taskname:
                _conll03get(dataset, directory, "de")
            elif "en" in taskname:
                _conll03get(dataset, directory, "en")
            else:
                logger.error("Cannot download {}. Unknown data source.".format(taskname))
    elif taskname not in DOWNSTREAM_TASK_MAP:
        logger.error("Cannot download {}. Unknown data source.".format(taskname))
    else:
        if os.name == "nt":  # make use of NamedTemporaryFile compatible with Windows
            delete_tmp_file = False
        else:
            delete_tmp_file = True
        with tempfile.NamedTemporaryFile(delete=delete_tmp_file) as temp_file:
            http_get(DOWNSTREAM_TASK_MAP[taskname], temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible

            # checking files for correctness with md5sum.
            if "germeval14" in taskname:
                if "2c9d5337d7a25b9a4bf6f5672dd091bc" != _get_md5checksum(temp_file.name):
                    logger.error(
                        f"Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py"
                    )
            elif "germeval18" in taskname:
                if "23244fa042dcc39e844635285c455205" != _get_md5checksum(temp_file.name):
                    logger.error(
                        f"Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py"
                    )
            elif "gnad" in taskname:
                if "ef62fe3f59c1ad54cf0271d8532b8f22" != _get_md5checksum(temp_file.name):
                    logger.error(
                        f"Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py"
                    )
            elif "germeval17" in taskname:
                if "f1bf67247dcfe7c3c919b7b20b3f736e" != _get_md5checksum(temp_file.name):
                    logger.error(
                        f"Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py"
                    )
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)
        # temp_file gets deleted here
