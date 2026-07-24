"""PyTorch Dataset (CLAUDE.md M3 / B.5 입력 명세).

각 keyframe을 끝점으로 과거 T=4 프레임 시퀀스를 조립한다(모델은 과거만 봄).
scene 경계에서는 앞을 패딩하고 마스크로 표시한다.

모델 입력 텐서:
  img_feat: (T, 6, D_f)  — 캐시된 백본 특징
  can:      (T, D_can)   — [speed,steering,ax,ay,yaw_rate, mask]
  risk:     (T, 6)       — 채널별 Σw(TTC), risk_source 조건에 따라 gt/noisy/none
  pad_mask: (T,)         — 1=유효 프레임, 0=패딩
학습 타깃(현재=마지막 프레임 기준):
  attention_view (int), layout (int), soft_label (6,), primary_view (int)

risk_source: gt | noisy | none  (+ 향후 lidar 훅). train_risk_noise=True면
학습 증강으로 매 아이템 노이즈 프리셋을 무작위 적용.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.noise import apply_noise, make_noise_config
from src.labeling.label_function import (
    LabelConfig, channel_risk_contributors, risk_vector_from_contribs,
)
from src.utils.constants import D_CAN, MANEUVER_INDEX, NUM_CAMERAS


class ViewSelectionDataset(Dataset):
    def __init__(self, wrapper, feature_cache, labels_df, label_cfg: LabelConfig,
                 scene_names: list[str], seq_len: int = 4,
                 risk_source: str = "gt", noise_preset: str = "clean",
                 train_risk_noise: bool = False, seed: int = 42):
        self.w = wrapper
        self.fc = feature_cache
        self.cfg = label_cfg
        self.seq_len = seq_len
        self.risk_source = risk_source
        self.noise_preset = noise_preset
        self.train_risk_noise = train_risk_noise
        self.feat_dim = feature_cache.feat_dim
        self._base_seed = seed

        name_set = set(scene_names)
        self.labels = labels_df.set_index("sample_token")

        # scene별 시간순 토큰 + 평탄 인덱스
        self.scene_tokens: dict[str, list[str]] = {}
        self.flat: list[tuple[str, int]] = []      # (scene_token, pos)
        for st in wrapper.scene_tokens():
            if wrapper.scene_name(st) not in name_set:
                continue
            toks = wrapper.sample_tokens_in_scene(st)
            toks = [t for t in toks if t in self.labels.index]
            if not toks:
                continue
            self.scene_tokens[st] = toks
            for pos in range(len(toks)):
                self.flat.append((st, pos))

        # 위험 기여 + CAN 특징 사전 계산(메모리 캐시 — mini 규모에 적합)
        self._contribs: dict[str, list] = {}
        self._can: dict[str, tuple] = {}
        for st, toks in self.scene_tokens.items():
            for t in toks:
                self._contribs[t] = channel_risk_contributors(wrapper, t, label_cfg)
                self._can[t] = wrapper.get_can_features(t)

        # 노이즈 파라미터(위험 재계산 정합)
        self._risk_params = dict(tau=label_cfg.tau, w_type=label_cfg.w_type,
                                 w_min_near=label_cfg.w_min_near,
                                 near_range=label_cfg.near_range)

    def __len__(self):
        return len(self.flat)

    # ------------------------------------------------------------------
    def _risk_and_blackout(self, token: str, rng: np.random.Generator):
        """risk_source 조건에 따른 (risk_vec[6], blackout 채널 목록)."""
        source = self.risk_source
        preset = self.noise_preset
        if self.train_risk_noise:                  # 학습 증강: 노이즈 강제
            source = "noisy"
            preset = str(rng.choice(["light", "medium"]))

        contribs = self._contribs[token]
        if source == "none":
            return np.zeros(NUM_CAMERAS, dtype=np.float32), []
        if source == "gt":
            return risk_vector_from_contribs(contribs).astype(np.float32), []
        if source == "noisy":
            nc = make_noise_config(preset, risk_params=self._risk_params)
            res = apply_noise(contribs, rng, nc)
            return risk_vector_from_contribs(res.contribs).astype(np.float32), res.blackout
        # 향후: source == "lidar" → LiDAR 검출기 contributors 사용
        raise ValueError(f"알 수 없는 risk_source: {source}")

    def __getitem__(self, i: int):
        scene_token, pos = self.flat[i]
        toks = self.scene_tokens[scene_token]
        rng = np.random.default_rng(self._base_seed + i)

        T = self.seq_len
        img = np.zeros((T, NUM_CAMERAS, self.feat_dim), dtype=np.float32)
        can = np.zeros((T, D_CAN), dtype=np.float32)
        risk = np.zeros((T, NUM_CAMERAS), dtype=np.float32)
        pad_mask = np.zeros((T,), dtype=np.float32)

        # 과거 T 프레임: [pos-T+1 .. pos], scene 시작 이전은 패딩
        for k in range(T):
            src_pos = pos - (T - 1 - k)
            if src_pos < 0:
                continue                            # 패딩(0), pad_mask=0 유지
            tok = toks[src_pos]
            pad_mask[k] = 1.0
            img[k] = self.fc.get(tok)
            feat, mask = self._can[tok]
            can[k, :len(feat)] = feat
            can[k, -1] = mask
            rv, blackout = self._risk_and_blackout(tok, rng)
            risk[k] = rv
            for ch in blackout:                     # 채널 블랙아웃: img_feat 0
                img[k, ch] = 0.0

        row = self.labels.loc[toks[pos]]
        soft = np.array([row[f"soft_{c}"] for c in range(NUM_CAMERAS)], dtype=np.float32)
        return {
            "img_feat": torch.from_numpy(img),
            "can": torch.from_numpy(can),
            "risk": torch.from_numpy(risk),
            "pad_mask": torch.from_numpy(pad_mask),
            "attention_view": int(row["attention_view"]),
            "primary_view": int(row["primary_view"]),
            "layout": int(row["layout"]),
            "soft_label": torch.from_numpy(soft),
            "maneuver": str(row["maneuver"]),
        }

    # ------------------------------------------------------------------
    def make_sampler(self) -> WeightedRandomSampler:
        """기동 상태 기준 균형 WeightedRandomSampler (클래스 불균형 완화)."""
        mans = [self.labels.loc[self.scene_tokens[st][pos]]["maneuver"]
                for st, pos in self.flat]
        freq = np.bincount([MANEUVER_INDEX[m] for m in mans],
                           minlength=len(MANEUVER_INDEX))
        weights = np.array([1.0 / max(1, freq[MANEUVER_INDEX[m]]) for m in mans],
                           dtype=np.float64)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def collate(batch: list[dict]) -> dict:
    """기본 콜레이트 (maneuver 문자열은 리스트로)."""
    out = {}
    for key in ("img_feat", "can", "risk", "pad_mask", "soft_label"):
        out[key] = torch.stack([b[key] for b in batch])
    for key in ("attention_view", "primary_view", "layout"):
        out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
    out["maneuver"] = [b["maneuver"] for b in batch]
    return out
