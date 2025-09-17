# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import copy

from transformers import LlamaConfig
from transformers import Swinv2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DiaNAConfig(PretrainedConfig):
    r"""
    [`DiaNAConfig`] is the configuration class to store the configuration of a
    [`DiaNAModel`]. It is used to instantiate a DiaNAModel according to the specified
    arguments, defining the Swinv2Model and LLaMA3 configs. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the DiaNAModel architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Swinv2Config`].
        llama_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`LLaMAConfig`].
        embed_dim (`int`, *optional*, defaults to 768):
            Size of the embeddings from the CLIP model.
        label_smoothing (`float`, *optional*, defaults to 0.0):
            The amount of label smoothing to apply.
        use_vision_lora (`int`, *optional*, defaults to 0):
            If non-zero, indicates the use of LoRA in the vision of the model.
        use_llama_lora (`int`, *optional*, defaults to 0):
            If non-zero, indicates the use of LoRA in the LLaMA of the model.
        force_image_size (`int` or `None`, *optional*):
            If not None, forces the model to use this specific image size.
        kwargs (*optional*):
            Dictionary of additional keyword arguments.
    """

    model_type = 'main'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            llama_config=None,
            refiner_config=None,
            embed_dim=768,
            label_smoothing=0.0,
            use_vision_lora=0,
            use_llama_lora=0,
            force_image_size=None,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing the config with default values (`Swinv2Config`).')

        if llama_config is None:
            llama_config = {}
            logger.info(
                'llama_config is None. Initializing the config with default values (`LlamaConfig`).')

        if refiner_config is None:
            refiner_config = {}
            logger.info(
                'refiner_config is None. Initializing the config with default values (`RefinerConfig`).')

        self.vision_config = Swinv2Config(**vision_config)
        self.llama_config = LlamaConfig(**llama_config)
        self.refiner_config = RefinerConfig(**refiner_config)
        self.hidden_size = self.llama_config.hidden_size

        self.embed_dim = embed_dim
        self.label_smoothing = label_smoothing
        self.use_vision_lora = use_vision_lora
        self.use_llama_lora = use_llama_lora
        self.force_image_size = force_image_size

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llama_config'] = self.llama_config.to_dict()
        output['refiner_config'] = self.refiner_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output


class RefinerConfig(PretrainedConfig):
    model_type = 'refiner'

    def __init__(
            self,
            enable: bool = False,
            num_query: int = 32,
            num_hidden_layers: int = 2,
            sim_type: str = 'avg',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable = enable
        self.num_query = num_query
        self.num_hidden_layers = num_hidden_layers
        self.sim_type = sim_type
