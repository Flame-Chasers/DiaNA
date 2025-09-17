from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel

from .configuration_diana import RefinerConfig
from .modeling_utils import GatherLayer
from .loss import compute_itc, compute_sdm


class Refiner(PreTrainedModel):
    config_class = RefinerConfig
    _keys_to_ignore_on_load_unexpected = [r"query_token"]
    _keys_to_ignore_on_load_missing = [r"attr_query_token"]

    def __init__(self, config: RefinerConfig):
        super(Refiner, self).__init__(config)

        # self.query_token = nn.Parameter(torch.empty(1, config.num_query, config.hidden_size))
        self.attr_query_token = nn.Parameter(torch.empty(1, config.num_query, config.hidden_size))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.hidden_size // 64,
            batch_first=True
        )
        self.transformer = Transformer(
            width=config.hidden_size,
            layers=config.num_hidden_layers,
            heads=config.hidden_size // 64
        )
        self.sim_type = config.sim_type

        self.enable_sdm = config.enable_sdm

        self.post_init()

    def post_init(self):
        scale = self.transformer.width ** -0.5
        attn_std = scale
        proj_std = scale * ((2 * self.transformer.layers) ** -0.5)
        fc_std = (2 * self.transformer.width) ** -0.5

        # inti query_token
        self.attr_query_token.data.normal_(std=0.02)

        # init cross_attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        # init transformer
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward_train(self, x, attention_mask=None):
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        x = self.cross_attn(
            query=self.attr_query_token.repeat(x.shape[0], 1, 1),
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x

    def compute_sim(self, attrs_embeds_1: torch.Tensor, attrs_embeds_2: torch.Tensor):
        """
        Compute cosine similarity for each attribute.

        Args:
            attrs_embeds_1: (N, K, C)
            attrs_embeds_2: (M, K, C)
            K is the number of attributes.

        Returns:
            sim: (N, M)
        """
        attrs_embeds_1 = attrs_embeds_1.unsqueeze(1)  # N 1 K C
        attrs_embeds_2 = attrs_embeds_2.unsqueeze(0)  # 1 M K C

        sim = F.cosine_similarity(attrs_embeds_1, attrs_embeds_2, dim=-1)

        if self.sim_type == 'avg':
            sim = sim.mean(dim=-1)
        elif self.sim_type == 'max':
            sim = sim.max(dim=-1)[0]
        elif self.sim_type == 'min':
            sim = sim.min(dim=-1)[0]
        elif self.sim_type == 'lsp_max':
            sim = torch.logsumexp(sim, dim=-1)
        elif self.sim_type == 'lsp_min':
            sim = -torch.logsumexp(-sim, dim=-1)
        else:
            raise ValueError(f"Unknown sim_type: {self.sim_type}")

        return sim

    def forward(self, image_embeds, text_embeds, attention_mask, sim_targets, logit_scale):
        image_attrs = self.forward_train(image_embeds, attention_mask=None)  # N K C
        text_attrs = self.forward_train(text_embeds, attention_mask=attention_mask)  # N K C
        image_attrs_all = GatherLayer.apply(image_attrs.contiguous()).flatten(0, 1)  # NG K C
        text_attrs_all = GatherLayer.apply(text_attrs.contiguous()).flatten(0, 1)  # NG k C

        sim_i2t = self.compute_sim(image_attrs, text_attrs_all)
        sim_t2i = self.compute_sim(text_attrs, image_attrs_all)

        sim_i2t = logit_scale * sim_i2t
        sim_t2i = logit_scale * sim_t2i

        loss_t2i = compute_itc(sim_t2i, sim_targets)
        loss_i2t = compute_itc(sim_i2t, sim_targets)
        loss = (loss_t2i + loss_i2t) / 2

        if self.enable_sdm:
            loss_t2i = compute_sdm(sim_t2i, sim_targets)
            loss_i2t = compute_sdm(sim_i2t, sim_targets)
            loss_sdm = (loss_t2i + loss_i2t) / 2
            loss = loss + loss_sdm

        return loss


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
