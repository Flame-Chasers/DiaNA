from transformers import AutoTokenizer
from transformers.models.auto.configuration_auto import replace_list_option_in_docstrings
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES

LLAMA3_PAD_TOKEN = "<|finetune_right_pad_id|>"
LLAMA3_MASK_TOKEN = "<|reserved_special_token_0|>"  # we use <|reserved_special_token_0|> as mask token
LLAMA3_SOLID_TOKEN_ID_UP_LIMIT = 128000


class Llama3Tokenizer(AutoTokenizer):

    @classmethod
    @replace_list_option_in_docstrings(TOKENIZER_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Manually setting the pad_token_id, which is by default set as None in the Llama3 tokenizer.
        if tokenizer.pad_token_id is None:
            pad_token_id = tokenizer.convert_tokens_to_ids(LLAMA3_PAD_TOKEN)
            tokenizer.pad_token_id = pad_token_id

        if tokenizer.mask_token_id is None:
            mask_token_id = tokenizer.convert_tokens_to_ids(LLAMA3_MASK_TOKEN)
            tokenizer.mask_token_id = mask_token_id
            tokenizer.mask_token = LLAMA3_MASK_TOKEN

        tokenizer.solid_token_id_up_limit = LLAMA3_SOLID_TOKEN_ID_UP_LIMIT

        return tokenizer
