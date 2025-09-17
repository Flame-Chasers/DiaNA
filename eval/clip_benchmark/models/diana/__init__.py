from .modeling_diana import DiaNAModel
from .configuration_diana import DiaNAConfig
from .llama3_tokenizer import Llama3Tokenizer

__all__ = ['DiaNAModel', 'DiaNAConfig', 'Llama3Tokenizer', 'load_diana']


def load_diana(pretrained, device):
    model = DiaNAModel.from_pretrained(pretrained)
    dtype = model.config.torch_dtype
    model = model.to(device=device, dtype=dtype)
    if model.config.use_llama_lora:
        model.llama.merge_and_unload()

    tokenizer = Llama3Tokenizer.from_pretrained(pretrained)
    return model, tokenizer

