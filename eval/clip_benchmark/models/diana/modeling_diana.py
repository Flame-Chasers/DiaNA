from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from timm.models.layers import trunc_normal_
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import ModelOutput

from .configuration_diana import DiaNAConfig
from .loss import compute_itc, compute_sdm
from .modeling_llama import LlamaModel
from .modeling_refiner import Refiner
from .modeling_swinv2 import Swinv2Model
from .modeling_utils import GatherLayer


@dataclass
class DiaNAModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InternVLModelOutput`].
    """

    loss: Optional[Union[torch.Tensor, torch.FloatTensor]] = None
    loss_itc: Optional[Union[torch.Tensor, torch.FloatTensor]] = None
    loss_sdm: Optional[Union[torch.Tensor, torch.FloatTensor]] = None
    loss_refine: Optional[Union[torch.Tensor, torch.FloatTensor]] = None
    logit_scale: Optional[Union[torch.Tensor, torch.FloatTensor]] = None


class DiaNAModel(PreTrainedModel):
    config_class = DiaNAConfig

    def __init__(self, config: DiaNAConfig):
        super(DiaNAModel, self).__init__(config)

        self.vision_model = Swinv2Model.from_pretrained(config.vision_config.pretrained, config=config.vision_config)
        self.llama = LlamaModel.from_pretrained(config.llama_config.pretrained, config=config.llama_config)

        self.vision_num_features = self.vision_model.num_features
        self.text_hidden_size = self.llama.hidden_size
        embed_dim = config.embed_dim

        self.vision_projection = nn.Linear(self.vision_num_features, embed_dim)
        self.text_projection = nn.Linear(self.text_hidden_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(config.logit_scale))  # trainable

        # refiner module
        self.enable_refiner = config.refiner_config.enable_refiner
        if self.enable_refiner:
            config.refiner_config.hidden_size = embed_dim
            self.refiner = Refiner(config.refiner_config)

        self.enable_sdm = config.enable_sdm

        self.label_smoothing = config.label_smoothing
        self.gradient_checkpointing = True

        # Initialize weights and apply final processing
        self.post_init()

        if config.use_llama_lora:
            self.wrap_llama_lora(r=config.use_llama_lora, lora_alpha=config.use_llama_lora * 2)

    def post_init(self):
        for module in (self.vision_projection, self.text_projection):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def wrap_llama_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.print_trainable_parameters()

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    @property
    def dtype(self) -> torch.dtype:
        return self.logit_scale.dtype

    def encode_image(self, pixel_values):
        pixel_values = pixel_values.to(self.device, self.dtype)
        image_output = self.vision_model(pixel_values)
        image_embeds, image_dense = image_output.pooler_output, image_output.last_hidden_state
        image_embeds = self.vision_projection(image_embeds)
        image_dense = self.vision_projection(image_dense)
        return image_embeds, image_dense

    def encode_text(self, input_ids, attention_mask):
        text_dense = self.llama(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

        selected = attention_mask.sum(1) - 1
        text_embeds = text_dense[torch.arange(text_dense.shape[0]), selected]
        text_embeds = self.text_projection(text_embeds)
        text_dense = self.text_projection(text_dense)
        return text_embeds, text_dense

    def refine(self, embeddings, attention_mask=None):
        return self.refiner.forward_train(embeddings, attention_mask)

    def refine_sim(self, attrs_embeds_1, attrs_embeds_2):
        return self.refiner.compute_sim(attrs_embeds_1, attrs_embeds_2)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            image_ids: torch.LongTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DiaNAModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, hidden_size)
        image_output = self.vision_model(pixel_values)
        image_embeds, image_dense = image_output.pooler_output, image_output.last_hidden_state
        image_embeds = self.vision_projection(image_embeds)
        image_dense = self.vision_projection(image_dense)

        # step 2: forward the input_ids and attention_mask through the text encoder.
        text_dense = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state

        ###============== Image-Text Contrastive ===================###
        selected = attention_mask.sum(1) - 1
        text_embeds = text_dense[torch.arange(text_dense.shape[0]), selected]
        text_embeds = self.text_projection(text_embeds)
        text_dense = self.text_projection(text_dense)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        image_embeds_all = GatherLayer.apply(image_embeds).flatten(0, 1)
        text_embeds_all = GatherLayer.apply(text_embeds).flatten(0, 1)

        # cosine similarity as logits
        logit_scale = torch.clamp(self.logit_scale, max=np.log(1. / 0.01)).exp()
        sim_i2t = logit_scale * (image_embeds @ text_embeds_all.t())
        sim_t2i = logit_scale * (text_embeds @ image_embeds_all.t())

        ids = image_ids.view(-1, 1)
        ids_all = GatherLayer.apply(ids).flatten(0, 1)
        pos_idx = torch.eq(ids, ids_all.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        loss_t2i = compute_itc(sim_t2i, sim_targets)
        loss_i2t = compute_itc(sim_i2t, sim_targets)
        loss_itc = (loss_t2i + loss_i2t) / 2

        loss = loss_itc

        loss_sdm = None
        if self.enable_sdm:
            loss_t2i = compute_sdm(sim_t2i, sim_targets)
            loss_i2t = compute_sdm(sim_i2t, sim_targets)
            loss_sdm = (loss_t2i + loss_i2t) / 2
            loss = loss + loss_sdm

        # refine the embeddings
        loss_refine = None
        if self.enable_refiner:
            loss_refine = self.refiner(image_dense, text_dense, attention_mask, sim_targets, logit_scale)
            loss = loss + loss_refine

        return DiaNAModelOutput(
            loss=loss,
            loss_itc=loss_itc,
            loss_sdm=loss_sdm,
            loss_refine=loss_refine,
            logit_scale=logit_scale,
        )
