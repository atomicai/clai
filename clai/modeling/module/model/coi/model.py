import os
import pathlib

from clai.modeling.module.model import base
from transformers import AutoConfig, AutoModel

# These are the names of the attributes in various model configs which refer to the number of dimensions
# in the output vectors
OUTPUT_DIM_NAMES = ["dim", "hidden_size", "d_model"]


class COIModel(base.LanguageModel, calling_name="coi"):
    def __init__(self):
        super(COIModel, self).__init__()
        self.model = None
        self.name = "coi"
        self.remote_name = None

    def save(self, where):
        pass

    @classmethod
    def load(cls, pretrained_model_name_or_path="cointegrated/rubert-tiny", language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("bert-base-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """

        coi = cls()
        if "farm_lm_name" in kwargs:
            coi.name = kwargs["farm_lm_name"]
        else:
            coi.remote_name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = pathlib.Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            coi_config = AutoConfig.from_pretrained(farm_lm_config)
            farm_lm_model = pathlib.Path(pretrained_model_name_or_path) / "language_model.bin"
            coi.model = AutoModel.from_pretrained(farm_lm_model, config=coi_config, **kwargs)
            coi.language = coi.model.config.language
        else:
            # Pytorch-transformer Style
            coi.model = AutoModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            coi.language = None
        return coi

    def forward(self, input_ids, segment_ids=None, padding_mask=None, **kwargs):
        """
        Perform the forward pass of the BERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.config.output_hidden_states:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False
