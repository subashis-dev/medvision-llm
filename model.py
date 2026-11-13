import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for fast transformer convergence."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


# =========================================================================
# 1. ConvNeXt-Style Visual Feature Extractor
# =========================================================================
class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with 7x7 depthwise conv, LayerNorm, and inverted 1x1 MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)  # Channels-first LayerNorm equivalent
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class ConvNeXtVisionEncoder(nn.Module):
    """Extracts spatial visual feature tokens from medical MRI / scans."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens  # 4x4 spatial grid = 16 visual tokens

        # Stem: 4x downsampling (e.g. 128x128 -> 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=4),
            nn.GroupNorm(1, 128),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
        )

        # Downsample to stage 2: 2x downsampling (32x32 -> 16x16)
        self.downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.GroupNorm(1, 256),
        )
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
        )

        # Adaptive pool down to 4x4 spatial grid
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(256, out_dim)
        self.norm = RMSNorm(out_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, out_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.downsample(feat)
        feat = self.stage2(feat)
        feat = self.pool(feat)  # [B, 256, 4, 4]

        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # [B, 16, 256]
        tokens = self.norm(self.proj(feat)) + self.pos_emb  # [B, 16, out_dim]
        return tokens


# =========================================================================
# 2. Causal Multi-Head Self-Attention & Transformer Blocks
# =========================================================================
class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention with Flash/SDPA support."""

    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention with causal masking
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Causal Transformer Block."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ffn: int = 1024):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.GELU(),
            nn.Linear(d_ffn, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =========================================================================
# 3. MedVision-LLM (Unified Vision-Language Architecture)
# =========================================================================
class MedVisionVLM(nn.Module):
    """Unified Medical Vision-Language Model with ConvNeXt Vision Stem & Causal Transformer SLM."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ffn: int = 1024,
        max_seq_len: int = 256,
        num_vis_tokens: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_vis_tokens = num_vis_tokens

        # 1. Vision Feature Extractor & Auxiliary Classifier
        self.vision_encoder = ConvNeXtVisionEncoder(in_channels=3, out_dim=d_model, num_tokens=num_vis_tokens)
        self.vis_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )

        # 2. Text Token & Positional Embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # 3. Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # 4. Output LM Head (Tied to token embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        images: [B, 3, 128, 128]
        input_ids: [B, T] (Text prompt + target tokens)
        targets: Optional [B, T]
        loss_mask: Optional [B, T] (1.0 on target tokens, 0.0 on prompt/pad)
        labels: Optional [B] (0..3 ground truth pathology class)
        """
        B, T = input_ids.shape
        vis_tokens = self.vision_encoder(images)  # [B, 16, d_model]

        # Auxiliary direct visual classification logits
        vis_pooled = vis_tokens.mean(dim=1)
        cls_logits = self.vis_classifier(vis_pooled)  # [B, 4]

        # Text embeddings + positional embeddings
        text_embeds = self.tok_embeddings(input_ids)  # [B, T, d_model]
        text_pos = self.pos_embeddings[:, :T, :]
        text_embeds = text_embeds + text_pos

        # Unified single-stream multimodal sequence
        # [ <vis_0> ... <vis_15>, <bos>, prompt..., target... ]
        x = torch.cat([vis_tokens, text_embeds], dim=1)  # [B, 16 + T, d_model]

        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # Slice out logits corresponding to text positions
        text_out = x[:, self.num_vis_tokens:, :]  # [B, T, d_model]
        logits = self.lm_head(text_out)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            shift_logits = logits.contiguous().view(-1, self.vocab_size)
            shift_targets = targets.contiguous().view(-1)
            
            if loss_mask is not None:
                ce_loss = F.cross_entropy(shift_logits, shift_targets, reduction='none')
                flat_mask = loss_mask.contiguous().view(-1)
                lm_loss = (ce_loss * flat_mask).sum() / (flat_mask.sum() + 1e-8)
            else:
                lm_loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=0)

            if labels is not None:
                vis_loss = F.cross_entropy(cls_logits, labels)
                loss = lm_loss + 1.0 * vis_loss
            else:
                loss = lm_loss

        return logits, cls_logits, loss

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 140,
        temperature: float = 0.3,
        eos_token_id: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generates clinical findings conditioned on the visual scan."""
        self.eval()
        B = image.shape[0]
        vis_tokens = self.vision_encoder(image)  # [B, 16, d_model]
        cls_logits = self.vis_classifier(vis_tokens.mean(dim=1))  # [B, 4]

        curr_ids = prompt_ids.clone()  # [B, T]

        for _ in range(max_new_tokens):
            T = curr_ids.shape[1]
            if T >= self.max_seq_len - self.num_vis_tokens:
                break
            text_embeds = self.tok_embeddings(curr_ids) + self.pos_embeddings[:, :T, :]
            x = torch.cat([vis_tokens, text_embeds], dim=1)

            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)

            next_token_logits = self.lm_head(x[:, -1, :]) / max(temperature, 1e-5)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return curr_ids, cls_logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MedVisionVLM(vocab_size=256, d_model=256, n_layers=6, n_heads=8, d_ffn=1024).to(device)
    print(f"MedVisionVLM Parameters: {model.get_parameter_count():,} on {device}")
    dummy_img = torch.randn(2, 3, 128, 128, device=device)
    dummy_text = torch.randint(0, 255, (2, 40), device=device)
    dummy_targets = torch.randint(0, 255, (2, 40), device=device)
    logits, loss = model(dummy_img, dummy_text, targets=dummy_targets)
    print("Forward pass successful! Logits:", logits.shape, "Loss:", loss.item())
