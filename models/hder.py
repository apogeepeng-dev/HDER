# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import math
import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
from .model_utils import MultiheadAttention
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

logger = logging.getLogger(__name__)

__all__ = [
    'HDER',
    'ChannelGate',
    'ResidualAdapter',
    'build_hder_model',
]

# logit_scale ：[0.01, 1.0]
LOGIT_SCALE_MIN = math.log(1.0)
LOGIT_SCALE_MAX = math.log(100.0)


# ═══════════════════════════════════════════════════════════════════════════
# 文本侧（与 hder.py 保持一致，未改动）
# ═══════════════════════════════════════════════════════════════════════════
class TextEncoder(nn.Module):
    """CONCH text encoder：encode prompt embeddings into text feature vectors (512-dim)。"""

    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, conch_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]          # 768

        self.tokenizer = get_tokenizer()
        self.context_length = int(conch_model.text.positional_embedding.shape[0])  # 128

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(self.tokenizer, [ctx_init])
            with torch.no_grad():
                embedding = conch_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [name for name in classnames]

        # Use CONCH tokenizer to encode class aliases
        try:
            tokenized_prompts = tokenize(self.tokenizer, prompts)
        except AttributeError:
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.context_length,
            )
            tokenized_prompts = enc["input_ids"]

        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = [
            len(self.tokenizer.encode(name, max_length=127, truncation=True))
            for name in classnames
        ]
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class ChannelGate(nn.Module):
    def __init__(self, channels, k_size=5, scale=0.1):
        super().__init__()
        assert k_size % 2 == 1, "k_size must be an odd number"
        self.scale = float(scale)
        self.k_size = int(k_size)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        # x: [N, C]
        desc = x.mean(dim=0, keepdim=True)                                  # [1, C] 
        s = torch.sigmoid(self.conv(desc.unsqueeze(1))).squeeze(1)          # [1, C] 
        return x + self.scale * x * (2.0 * s - 1.0)                         # [N, C]

class ResidualAdapter(nn.Module):
    def __init__(self, dim, hidden=None, dropout=0.1):
        super().__init__()
        hidden = hidden or max(dim // 4, 64)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


# ═══════════════════════════════════════════════════════════════════════════
# model
# ═══════════════════════════════════════════════════════════════════════════
class HDER(nn.Module):


    def __init__(self, config, num_classes=3):
        super(ViLa_MIL_Model, self).__init__()
        self.num_classes = num_classes
        self.L = int(config.input_size)
        self.D = int(config.hidden_size)
        self.K = 1
        self.dropout = float(getattr(config, 'dropout', 0.1))
        self.gate_scale = float(getattr(config, 'gate_scale', 0.1))
        self.eca_k = int(getattr(config, 'eca_k', 5))
        self.logit_mode = str(getattr(config, 'logit_mode', 'cosine'))
        self.use_patch_norm = bool(getattr(config, 'use_patch_norm', False))
        label_smoothing = float(getattr(config, 'label_smoothing', 0.1))

        if getattr(config, 'prototype_number', None) is None:
            raise ValueError("config.prototype_number cannot be empty")
        if self.logit_mode not in ('cosine', 'dot'):
            raise ValueError(f"logit_mode only supports' cosine '/' dot ', received {self.logit_mode!r}")

        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ── load CONCH ──
        conch_model, _ = create_model_from_pretrained('conch_ViT-B-16', 'ckpts/conch.pth')
        conch_model = conch_model.float()

        self._prune_visual = bool(getattr(config, 'prune_visual', True))
        if self._prune_visual and hasattr(conch_model, 'visual'):
            del conch_model.visual
            logger.info('hder: The CONCH visual backbone has been removed (not involved in forward direction, reducing checkpoint/video memory)')

        self._freeze_text = bool(getattr(config, 'freeze_text_encoder', True))
        if self._freeze_text:
            for p in conch_model.text.parameters():
                p.requires_grad_(False)
            conch_model.eval()
            logger.info('hder: the CONCH text tower has been frozen, only training prompt ctx and image side')

        self.conch_model = conch_model
        self.tokenizer = get_tokenizer()

        self.prompt_learner = PromptLearner(config.text_prompt, conch_model)
        self.text_encoder = TextEncoder(conch_model)
        self.text_projector = nn.Identity()

    
        self.visual_adapter = ResidualAdapter(self.L, dropout=self.dropout)
        self.channel_gate = ChannelGate(self.L, k_size=self.eca_k, scale=self.gate_scale)
        self.token_gate = TokenGate(self.L, scale=self.gate_scale)
        self.norm_patch = nn.LayerNorm(self.L) if self.use_patch_norm else nn.Identity()
        self.drop_out = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.L)
        self.cross_attention_1 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        self.learnable_image_center = nn.Parameter(
            torch.empty(int(config.prototype_number), 1, self.L)
        )
        trunc_normal_(self.learnable_image_center, std=.02)

        # ── ABMIL Prototype Aggregation ──
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.scale_weights = nn.Parameter(torch.zeros(2))


        self.class_names = [name.replace('_', ' ') for name in config.text_prompt[:num_classes]]

    def train(self, mode=True):
        super(ViLa_MIL_Model, self).train(mode)
        if getattr(self, '_freeze_text', False) and isinstance(getattr(self, 'conch_model', None), nn.Module):
            self.conch_model.eval()
        return self

    def _text_features(self):
        prompts = self.prompt_learner()
        text = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        return self.text_projector(text.float())

    def _refine_patches(self, x):
        """patch level refine：[N, C] -> [N, C]"""
        M = self.visual_adapter(x)
        M = self.channel_gate(M)
        M = self.token_gate(M)
        return self.norm_patch(M)

    def _prototype_weights(self, proto):
        """ABMIL gateAttention：[K, C] -> [K]"""
        A_V = self.attention_V(proto)
        A_U = self.attention_U(proto)
        A = self.attention_weights(A_V * A_U)          # [K, 1]
        A = torch.transpose(A, 1, 0)                   # [1, K]
        return F.softmax(A, dim=1).squeeze(0)          # [K]

    def _encode_scale(self, x, need_attn=False):
        x = x.float()
        if x.dim() == 3:
            x = x.squeeze(0)                      # [1, N, C] -> [N, C]
        if x.dim() != 2:
            raise ValueError(f'patch feature should be [N, C] or [1, N, C], actually {tuple(x.shape)}')

        M = self._refine_patches(x)

        if need_attn:
            compents, raw = self.cross_attention_1(
                self.learnable_image_center, M, M, need_weights=True, need_raw=True)
        else:
            compents, _ = self.cross_attention_1(
                self.learnable_image_center, M, M)
            raw = None

        compents = self.norm(compents + self.learnable_image_center)   # [K, 1, C]
        proto = self.drop_out(compents.squeeze(1))                     # [K, C]

        A = self._prototype_weights(proto)                             # [K]
        feats = torch.mm(A.unsqueeze(0), proto)                        # [1, C]

        attn = None
        if raw is not None:
            raw = raw.mean(dim=1) if raw.dim() == 4 else raw            # [1, K, N]
            attn = F.softmax(raw.squeeze(0), dim=-1)                   # [K, N]

        return {
            'features': feats,
            'prototypes': proto,
            'patches': M,
            'prototype_weights': A,
            'attn': attn,
        }

    def _refine_text(self, txt, enc):
        """Text decoding：return [n_cls, C]。"""
        ctx = torch.cat((enc['prototypes'], enc['patches']), dim=0)     # [K + N, C]
        out, _ = self.cross_attention_2(txt.unsqueeze(1), ctx, ctx)     # [n_cls, 1, C]
        return out.squeeze(1) + txt                                     # [n_cls, C]

    def _logits_one_scale(self, img, txt):
        """[1, C] x [n, C] -> [1, n]"""
        if self.logit_mode == 'cosine':
            img = F.normalize(img, dim=-1)
            txt = F.normalize(txt, dim=-1)
            scale = self.logit_scale.clamp(min=LOGIT_SCALE_MIN, max=LOGIT_SCALE_MAX).exp()
            return scale * torch.mm(img, txt.t())
        return torch.mm(img, txt.t())

    def _fuse_logits(self, logits_low, logits_high):
        w = F.softmax(self.scale_weights, dim=0)                        # [2]
        return w[0] * logits_low + w[1] * logits_high

    def _infer(self, x_s, x_l, need_attn=False):
        text = self._text_features()                                    # [2n, C]
        low = self._encode_scale(x_s)
        high = self._encode_scale(x_l, need_attn=need_attn)

        txt_low = self._refine_text(text[:self.num_classes], low)
        txt_high = self._refine_text(text[self.num_classes:], high)

        logits_low = self._logits_one_scale(low['features'], txt_low)   # [1, n]
        logits_high = self._logits_one_scale(high['features'], txt_high)
        logits = self._fuse_logits(logits_low, logits_high)             # [1, n]

        return {
            'logits': logits,
            'low': low,
            'high': high,
            'text': text,
            'txt_low': txt_low,
            'txt_high': txt_high,
            'logits_low': logits_low,
            'logits_high': logits_high,
        }

    # ═══════════════════════════════════════════════════════════════════
    # traning / inference interface
    # ═══════════════════════════════════════════════════════════════════
    def forward(self, x_s, coord_s, x_l, coords_l, label):
        out = self._infer(x_s, x_l)
        logits = out['logits']

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return Y_prob, Y_hat, loss

    # ═══════════════════════════════════════════════════════════════════
    # Visualization
    # ═══════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def extract_slide_features(self, x_s, coord_s, x_l, coords_l):

        out = self._infer(x_s, x_l)
        feat = (out['low']['features'] + out['high']['features']) / 2.0
        if self.logit_mode == 'cosine':
            feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def get_attention_weights(self, x_l):
        """
        Returns:
            patch_scores:      [N]     归一化到 [0, 1]
            attn_proto2patch:  [K, N]
            A_proto:           [K]
        """
        enc = self._encode_scale(x_l, need_attn=True)
        if enc['attn'] is None:
            raise RuntimeError('Attention matrix for cross attention_1 not obtained')

        A = enc['prototype_weights']                        # [K]
        scores = torch.matmul(A, enc['attn'])               # [N]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores, enc['attn'], A

    @torch.no_grad()
    def get_refined_prototypes(self, x_s, x_l):
        """
        Returns:
            prototypes_low:  [K, D]
            prototypes_high: [K, D]
        """
        low = self._encode_scale(x_s)
        high = self._encode_scale(x_l)
        return low['prototypes'], high['prototypes']

    @torch.no_grad()
    def encode_tissue_descriptions(self, tissue_descriptions):
        """
        Returns:
            tissue_features: [num_tissues, D]  L2 归一化后的文本特征
        """
        device = next(self.parameters()).device
        tokenized = tokenize(self.tokenizer, tissue_descriptions).to(device)
        text_features = self.conch_model.encode_text(tokenized)
        return F.normalize(text_features, dim=-1)

    


# ═══════════════════════════════════════════════════════════════════════════
# build model
# ═══════════════════════════════════════════════════════════════════════════
def build_hder_model(args):
    import ml_collections
    config = ml_collections.ConfigDict()
    config.input_size = 512            # CONCH ViT-B/16
    config.hidden_size = 192           # ABMIL 
    config.text_prompt = args.text_prompt
    config.prototype_number = args.prototype_number
    config.dropout = 0.25 if getattr(args, 'drop_out', False) else 0.1
    config.gate_scale = float(getattr(args, 'gate_scale', 0.1))
    config.eca_k = int(getattr(args, 'eca_k', 5))
    config.logit_mode = str(getattr(args, 'logit_mode', 'cosine'))
    config.use_patch_norm = bool(getattr(args, 'patch_norm', False))
    config.freeze_text_encoder = not bool(getattr(args, 'no_freeze_text', False))
    config.prune_visual = not bool(getattr(args, 'keep_visual', False))
    model = ViLa_MIL_Model(config=config, num_classes=args.n_classes)
    return model

