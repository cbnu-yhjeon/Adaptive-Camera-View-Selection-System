"""융합 모델 (CLAUDE.md B.5) — 제안 모델 + ablation/B3/B4 공용.

구조:
  img_feat(B,T,6,D_f) → Linear → 채널 토큰 6개 (d)
    + camera positional emb(6)
    + risk(B,T,6) → Linear(1→d) 채널별 가산       [use_risk]
  can(B,T,d_can) → Linear → CAN 토큰 1개            [use_can]
  [6 채널 + 1 CAN] → TransformerEncoder(2층,4헤드)   [use_transformer]
    (off이면 채널별 독립 MLP + CAN 가산 — cross-view 제거, A4)
  → 채널 토큰 6개 (B,T,6,d)
  → 채널별 GRU 시간축 → 마지막 스텝 (B,6,d)          [use_gru; off면 마지막 프레임]
  → Head1: 채널 스칼라 → **채널별 독립 로짓** (6) → sigmoid = 강조 확률
  → Head2: 채널 mean-pool → MLP → layout (5-way)

Head1을 softmax로 묶지 않는 이유: 뷰 정책이 보조 뷰 0~2장을 고르는 다중 선택이다.
6채널 중 하나를 고르는 softmax로는 0장도 2장도 표현할 수 없고, 추론 임계값을 쓸어
(표시량 vs 안전) 운영 곡선을 그릴 수도 없다. 진행 방향 채널은 상시 표시이므로
손실에서 마스크로 제외한다 — 타깃 0으로 두면 "앞은 끄는 것"을 배운다.

플래그 조합: 제안(전부 on) / B3(use_can=use_risk=False) /
B4(use_gru=False,T=1) / A1(use_can=False) / A2(use_risk=False) /
A3(use_gru=False) / A4(use_transformer=False).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import D_CAN, NUM_CAMERAS, NUM_LAYOUTS


@dataclass
class ModelConfig:
    feat_dim: int = 576
    d: int = 128
    heads: int = 4
    ff: int = 256
    tf_layers: int = 2
    d_can: int = D_CAN
    num_layout: int = NUM_LAYOUTS
    dropout: float = 0.1
    use_can: bool = True
    use_risk: bool = True
    use_transformer: bool = True
    use_gru: bool = True


class ViewSelector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.img_proj = nn.Linear(cfg.feat_dim, d)
        self.cam_emb = nn.Parameter(torch.zeros(NUM_CAMERAS, d))
        nn.init.normal_(self.cam_emb, std=0.02)
        if cfg.use_risk:
            self.risk_proj = nn.Linear(1, d)
        if cfg.use_can:
            self.can_proj = nn.Linear(cfg.d_can, d)

        if cfg.use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.heads, dim_feedforward=cfg.ff,
                dropout=cfg.dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.tf_layers)
        else:  # A4: cross-view 제거 → 채널별 독립 MLP
            self.channel_mlp = nn.Sequential(
                nn.Linear(d, cfg.ff), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.ff, d))

        if cfg.use_gru:
            self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)

        self.head_attn = nn.Linear(d, 1)                      # 채널별 스칼라
        self.head_layout = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.num_layout))

    # ------------------------------------------------------------------
    def _frame_tokens(self, img_feat, can, risk):
        """프레임별 채널 표현 (B,T,6,d) — 토큰화 + (Transformer or MLP)."""
        B, T = img_feat.shape[:2]
        d = self.cfg.d
        ch = self.img_proj(img_feat)                          # (B,T,6,d)
        ch = ch + self.cam_emb.view(1, 1, NUM_CAMERAS, d)
        if self.cfg.use_risk:
            ch = ch + self.risk_proj(risk.unsqueeze(-1))      # (B,T,6,d)

        can_tok = None
        if self.cfg.use_can:
            can_tok = self.can_proj(can).unsqueeze(2)         # (B,T,1,d)

        if self.cfg.use_transformer:
            tokens = ch if can_tok is None else torch.cat([ch, can_tok], dim=2)
            n = tokens.shape[2]
            tokens = tokens.reshape(B * T, n, d)
            out = self.encoder(tokens).reshape(B, T, n, d)
            return out[:, :, :NUM_CAMERAS, :]                 # 채널 토큰 6개
        # cross-view 없음: 채널별 MLP + (CAN 가산으로만 융합)
        ch = self.channel_mlp(ch)
        if can_tok is not None:
            ch = ch + can_tok                                 # 브로드캐스트 (B,T,6,d)
        return ch

    def forward(self, batch: dict) -> dict:
        img_feat = batch["img_feat"]                          # (B,T,6,D_f)
        can = batch["can"]
        risk = batch["risk"]
        B, T = img_feat.shape[:2]
        d = self.cfg.d

        frames = self._frame_tokens(img_feat, can, risk)      # (B,T,6,d)

        if self.cfg.use_gru:
            seq = frames.permute(0, 2, 1, 3).reshape(B * NUM_CAMERAS, T, d)
            out, _ = self.gru(seq)                            # (B*6,T,d)
            out = out.reshape(B, NUM_CAMERAS, T, d).permute(0, 2, 1, 3)  # (B,T,6,d)
            final = out[:, -1]                                # (B,6,d) 마지막 스텝
            per_frame = out                                   # 시간 일관성용
        else:
            final = frames[:, -1]                             # (B,6,d)
            per_frame = frames

        attn_logits = self.head_attn(final).squeeze(-1)       # (B,6) 채널별 독립 로짓
        layout_logits = self.head_layout(final.mean(dim=1))   # (B,5)
        # 프레임별 강조 확률 (temporal consistency 손실용).
        # 채널별 sigmoid다 — softmax로 묶으면 "6채널 중 하나"가 되어 0장·2장을
        # 표현할 수 없고, 추론 임계값을 쓸어 운영 곡선을 그릴 수도 없다.
        attn_seq = torch.sigmoid(self.head_attn(per_frame).squeeze(-1))  # (B,T,6)
        return {"attn_logits": attn_logits, "layout_logits": layout_logits,
                "attn_seq": attn_seq}


# ---------------------------------------------------------------------
# 손실
# ---------------------------------------------------------------------
def focal_ce(logits, target, gamma: float = 2.0):
    """focal cross-entropy (클래스 불균형) — 단일 선택 비교용으로 유지."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
    return (-((1 - p_t) ** gamma) * logp_t).mean()


def focal_bce(logits, target, mask, gamma: float = 2.0, pos_weight: float = 1.0):
    """채널별 focal BCE — 다중 선택(0~2장) 학습의 주 손실.

    강조 채널은 전체 채널 슬롯의 8% 남짓이라 그냥 BCE를 걸면 전부 0으로 예측하는 해에
    빠진다. focal 항으로 쉬운 음성의 기여를 낮추고, pos_weight로 양성을 끌어올린다.

    mask=0인 채널(진행 방향)은 손실에서 제외한다 — 선택 대상이 아니므로 그 채널의
    출력을 0으로 학습시키면 모델이 "앞은 끄는 것"이라는 잘못된 규칙을 배운다.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    w = target * pos_weight + (1 - target)
    loss = ((1 - p_t) ** gamma) * ce * w * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def kl_soft(logits, soft_target, eps: float = 1e-8):
    """KL(soft_target ‖ softmax(logits)) — soft_target>0 항만."""
    logp = F.log_softmax(logits, dim=-1)
    t = soft_target.clamp_min(0)
    return (t * ((t + eps).log() - logp)).sum(dim=-1).mean()


def temporal_consistency(attn_seq, pad_mask):
    """β항: Σ_t ‖p_t − p_{t−1}‖² (유효 프레임 쌍만)."""
    if attn_seq.shape[1] < 2:
        return attn_seq.new_zeros(())
    diff = (attn_seq[:, 1:] - attn_seq[:, :-1]).pow(2).sum(dim=-1)   # (B,T-1)
    valid = (pad_mask[:, 1:] * pad_mask[:, :-1])                     # (B,T-1)
    denom = valid.sum().clamp_min(1.0)
    return (diff * valid).sum() / denom


@dataclass
class LossWeights:
    alpha: float = 0.5      # KL
    gamma_layout: float = 0.3
    beta: float = 0.1       # temporal consistency
    focal_gamma: float = 2.0
    pos_weight: float = 4.0  # 강조 채널이 희소해 양성 가중


def compute_loss(out: dict, batch: dict, lw: LossWeights) -> dict:
    l_attn = focal_bce(out["attn_logits"], batch["aux_target"], batch["aux_mask"],
                       lw.focal_gamma, lw.pos_weight)
    l_kl = kl_soft(out["attn_logits"], batch["soft_label"])
    l_layout = F.cross_entropy(out["layout_logits"], batch["layout"])
    l_temp = temporal_consistency(out["attn_seq"], batch["pad_mask"])
    total = l_attn + lw.alpha * l_kl + lw.gamma_layout * l_layout + lw.beta * l_temp
    return {"total": total, "attn": l_attn, "kl": l_kl,
            "layout": l_layout, "temporal": l_temp}


def select_views(logits, primary, threshold: float = 0.5, aux_max: int = 2):
    """추론 — 채널별 확률에 임계값을 걸어 보조 뷰 0~aux_max개를 고른다.

    threshold를 쓸면 (평균 보조 뷰 수 vs 위험 포착률) 운영 곡선이 그려진다.
    진행 방향은 상시 표시이므로 후보에서 뺀다.
    """
    p = torch.sigmoid(logits).clone()                      # (B,6)
    p.scatter_(1, primary.unsqueeze(1), -1.0)              # primary 제외
    keep = p >= threshold
    if aux_max is not None and aux_max < p.shape[1]:
        # 상위 aux_max개만 남긴다
        idx = p.argsort(dim=1, descending=True)[:, aux_max:]
        keep.scatter_(1, idx, False)
    return keep
