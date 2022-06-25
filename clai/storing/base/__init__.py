import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

import mmh3
import numpy as np


class Document:
    def __init__(
        self,
        text: str,
        id: Optional[str] = None,
        score: Optional[float] = None,
        probability: Optional[float] = None,
        question: Optional[str] = None,
        meta: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None,
        id_hash_keys: Optional[List[str]] = None,
        uuid_type: Optional[str] = None,
    ):
        """

        Note: There can be multiple Documents originating from one file (e.g. PDF), if you split the text
        into smaller passages. We'll have one Document per passage in this case.

        Each document has a unique ID. This can be supplied by the user or generated automatically.
        It's particularly helpful for handling of duplicates and referencing documents in other objects (e.g. Labels)

        There's an easy option to convert from/to dicts via `from_dict()` and `to_dict`.

        :param text: Text of the document
        :param id: Unique ID for the document. If not supplied by the user, we'll generate one automatically by
                   creating a hash from the supplied text. This behaviour can be further adjusted by `id_hash_keys`.
        :param score: Retriever's query score for a retrieved document
        :param probability: a pseudo probability by scaling score in the range 0 to 1
        :param question: Question text (e.g. for FAQs where one document usually consists of one question and one answer text).
        :param meta: Meta fields for a document like name, url, or author.
        :param embedding: Vector encoding of the text
        :param id_hash_keys: Generate the document id from a custom list of strings.
                             If you want ensure you don't have duplicate documents in your DocumentStore but texts are
                             not unique, you can provide custom strings here that will be used (e.g. ["filename_xy", "text_of_doc"].
        """

        self.text = text
        self.score = score
        self.probability = probability
        self.question = question
        self.meta = meta or {}
        self.embedding = embedding

        self.id = self._get_id(id_hash_keys, uuid_type=uuid_type) if id is None else str(id)

    def _get_id(self, id_hash_keys, uuid_type=None):
        if uuid_type is None:
            return "{:02x}".format(mmh3.hash128(self.text, signed=False))
        elif uuid_type == "uuid3":
            return str(uuid.uuid3(uuid.NAMESPACE_DNS, self.text))
        elif uuid_type == 'uuid5':
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, self.text))
        else:
            raise ValueError(f"Choose either \"uuid3\" or \"uuid5\" or None")

    def to_dict(self, field_map={}):
        inv_field_map = {v: k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, dict, field_map={}, uuid_type=None):
        _doc = deepcopy(dict)
        init_args = [
            "text",
            "id",
            "score",
            "probability",
            "question",
            "meta",
            "embedding",
        ]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        # copy additional fields into "meta"
        for k, v in _doc.items():
            if k not in init_args and k not in field_map:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _new_doc = {}
        for k, v in _doc.items():
            if k in init_args:
                _new_doc[k] = v
            elif k in field_map:
                k = field_map[k]
                _new_doc[k] = v

        if uuid_type:
            _new_doc["uuid_type"] = uuid_type

        return cls(**_new_doc)

    def __repr__(self):
        return str(self.to_dict())
