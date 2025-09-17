from typing import Dict

import transformers
from transformers.trainer_pt_utils import get_model_param_count


def format_numel_str(numel: int) -> str:
    B = 1000 ** 3
    M = 1000 ** 2
    K = 1000
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def print_model_param_info(model):
    total_n_parameters = get_model_param_count(model, trainable_only=False)
    print(f'number of total params: {format_numel_str(total_n_parameters)}')

    n_parameters = get_model_param_count(model, trainable_only=True)
    print(f'number of params with requires_grad: {format_numel_str(n_parameters)}')

    if hasattr(model, 'vision_model'):
        total_visual_n_parameters = get_model_param_count(model.vision_model, trainable_only=False)
        print(f'number of visual params: {format_numel_str(total_visual_n_parameters)}')
    if hasattr(model, 'llama'):
        total_text_n_parameters = get_model_param_count(model.llama, trainable_only=False)
        print(f'number of text params: {format_numel_str(total_text_n_parameters)}')


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
