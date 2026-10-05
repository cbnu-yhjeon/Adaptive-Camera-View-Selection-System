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

노이즈 난수 정책 (원칙 2·4)
--------------------------
- **학습 증강**: 같은 샘플이라도 epoch마다 다른 노이즈 실현을 봐야 한다. 그렇지
  않으면 고정된 손상 데이터를 암기하게 되어 강건성(M7)이 증명되지 않는다.
  시드는 (seed, epoch, index)의 결정적 함수이므로 재현성은 유지된다.
  학습 루프가 epoch마다 `set_epoch()`를 호출해야 한다.
- **평가**: (seed, index)만 사용해 아이템별로 고정한다. 제안 모델과 B2가
  **동일한 손상 입력**을 보아야 열화 곡선 비교가 성립하기 때문이다.
- 검출 품질(프리셋)과 카메라 결손(블랙아웃)은 2초 시퀀스 안에서 유지되는 성질이므로
  프레임마다 다시 뽑지 않고 **아이템 단위로 한 번** 정한다.
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


def item_rng(base_seed: int, index: int, epoch: int | None) -> np.random.Generator:
    """아이템용 난수 생성기.

    epoch=None이면 (seed, index)만으로 결정 → 실행·모델에 무관하게 항상 같은 노이즈.
    epoch가 주어지면 epoch마다 다른 실현이 나오되 (seed, epoch, index)로 재현 가능하다.
    """
    if epoch is None:
        return np.random.default_rng([base_seed, index])
    return np.random.default_rng([base_seed, epoch, index])


_FRAME_SALT = 0x46524D   # 'FRM' — item_rng의 [seed, epoch, index]와 겹치지 않는 세 번째 값
_SCENE_SALT = 0x53434E   # 'SCN' — 평가용 scene 단위 카메라 고장

# 강건성 실험(M7)의 유형별 노이즈. "강도:유형" 프리셋은 그 유형의 파라미터만 남긴다.
NOISE_TYPES = {
    "miss": ("p_miss",),
    "fp": ("lambda_fp",),
    "pos": ("sigma_pos", "sigma_ttc_rel"),
    "drop": (),                         # 검출 잡음 없음 — 카메라 고장만
}
_DETECTION_FIELDS = ("p_miss", "lambda_fp", "sigma_pos", "sigma_ttc_rel")
DROP_CAMERAS = {"clean": 0, "light": 1, "medium": 2, "heavy": 3}


def split_preset(preset: str) -> tuple[str, str | None]:
    """'medium' → ('medium', None),  'medium:fp' → ('medium', 'fp')."""
    if ":" in preset:
        level, typ = preset.split(":", 1)
        if typ not in NOISE_TYPES:
            raise ValueError(f"알 수 없는 노이즈 유형: {typ}")
        return level, typ
    return preset, None


def frame_rng(base_seed: int, frame_index: int) -> np.random.Generator:
    """평가용 **프레임** 난수 생성기 — 한 프레임의 검출 결과는 하나다.

    아이템 단위로 노이즈를 뽑으면 같은 프레임이 어느 2초 창에 들어가느냐에 따라 다르게
    손상된다. 학습 증강에는 그 다양성이 도움이 되지만, 평가에서는 "그 순간의 검출기 출력"이
    하나여야 한다. 그래야 전체 이력을 쓰는 규칙(B2)과 4프레임을 보는 모델이 같은 세계를 본다.
    """
    return np.random.default_rng([base_seed, frame_index, _FRAME_SALT])


class ViewSelectionDataset(Dataset):
    def __init__(self, wrapper, feature_cache, labels_df, label_cfg: LabelConfig,
                 scene_names: list[str], seq_len: int = 4,
                 risk_source: str = "gt", noise_preset: str = "clean",
                 train_risk_noise: bool = False, seed: int = 42,
                 contrib_cache=None,
                 train_noise_mix: tuple[str, ...] = ("light", "medium")):
        self.w = wrapper
        self.fc = feature_cache
        self.cfg = label_cfg
        self.seq_len = seq_len
        self.risk_source = risk_source
        self.noise_preset = noise_preset
        self.train_risk_noise = train_risk_noise
        # 학습 증강에서 아이템마다 고를 검출 품질. clean을 넣으면 깨끗한 입력도 보게 되고,
        # heavy를 넣으면 심한 손상도 본다 — light·medium만 쓰면 평가 clean에서 규칙보다
        # 6%p 낮았다(모델이 깨끗한 입력을 본 적이 없다).
        self.train_noise_mix = tuple(train_noise_mix)
        self.feat_dim = feature_cache.feat_dim
        self._base_seed = seed
        self._epoch = 0

        name_set = set(scene_names)
        self.labels = labels_df.set_index("sample_token")

        # scene별 시간순 토큰 + 평탄 인덱스
        self.scene_tokens: dict[str, list[str]] = {}
        self.flat: list[tuple[str, int]] = []      # (scene_token, pos)
        self._frame_idx: dict[str, int] = {}        # 토큰 → 그 토큰이 현재 프레임인 평탄 인덱스
        self._frame_scene: dict[str, str] = {}      # 토큰 → scene
        self._scene_idx: dict[str, int] = {}        # scene → 고정 번호 (scene 단위 난수용)
        for st in wrapper.scene_tokens():
            if wrapper.scene_name(st) not in name_set:
                continue
            toks = wrapper.sample_tokens_in_scene(st)
            toks = [t for t in toks if t in self.labels.index]
            if not toks:
                continue
            self.scene_tokens[st] = toks
            self._scene_idx[st] = len(self._scene_idx)
            for pos in range(len(toks)):
                self._frame_idx[toks[pos]] = len(self.flat)
                self._frame_scene[toks[pos]] = st
                self.flat.append((st, pos))

        # 위험 기여 + CAN 특징. 디스크 캐시(contrib_cache.ContribCache)가 주어지면
        # 그대로 쓰고, 없으면 메모리에 직접 계산한다(mini 규모 스모크 테스트용).
        self._cache = contrib_cache
        self._contribs: dict[str, list] = {}
        self._can: dict[str, tuple] = {}
        if self._cache is None:
            for st, toks in self.scene_tokens.items():
                for t in toks:
                    self._contribs[t] = channel_risk_contributors(wrapper, t, label_cfg)
                    self._can[t] = wrapper.get_can_features(t)

        # 노이즈 파라미터(위험 재계산 정합)
        self._risk_params = dict(tau=label_cfg.tau, w_type=label_cfg.w_type,
                                 w_min_near=label_cfg.w_min_near,
                                 near_range=label_cfg.near_range)
        self._nc: dict[str, object] = {}     # 프리셋별 NoiseConfig 재사용
        self._frame_risk: dict[tuple, np.ndarray] = {}   # 평가용 프레임 위험 메모

    def __len__(self):
        return len(self.flat)

    def set_epoch(self, epoch: int) -> None:
        """학습 루프가 epoch마다 호출한다 — 증강 노이즈를 매 epoch 새로 뽑기 위함.

        DataLoader 워커는 epoch마다 다시 포크되므로, 이터레이터를 만들기 전에
        호출해야 워커에 값이 전달된다.
        """
        self._epoch = int(epoch)

    # ------------------------------------------------------------------
    def _get_contribs(self, token: str) -> list[dict]:
        return self._cache.get(token) if self._cache is not None else self._contribs[token]

    def _get_can(self, token: str):
        return self._cache.can(token) if self._cache is not None else self._can[token]

    def _noise_config(self, preset: str):
        """프리셋별 NoiseConfig. 블랙아웃은 따로 다루므로 여기선 끈다.

        "강도:유형"이면 그 유형의 검출 잡음만 남기고 나머지는 0으로 둔다.
        """
        if preset not in self._nc:
            level, typ = split_preset(preset)
            over = {"n_blackout": 0}
            if typ is not None:
                keep = NOISE_TYPES[typ]
                over.update({f: 0.0 for f in _DETECTION_FIELDS if f not in keep})
            self._nc[preset] = make_noise_config(level, risk_params=self._risk_params, **over)
        return self._nc[preset]

    def _item_noise(self, rng: np.random.Generator):
        """아이템 전체에 고정되는 노이즈 조건 → (source, preset, blackout 채널).

        검출 품질과 카메라 결손은 2초 시퀀스 내내 유지되는 성질이라 프레임마다
        다시 뽑지 않는다.
        """
        source = self.risk_source
        preset = self.noise_preset
        if self.train_risk_noise:                  # 학습 증강: 노이즈 강제
            source = "noisy"
            preset = str(rng.choice(list(self.train_noise_mix)))

        blackout: list[int] = []
        if source == "noisy":
            n_blk, _ = self._blackout_spec(preset)
            if n_blk > 0:
                blackout = [int(c) for c in rng.choice(
                    NUM_CAMERAS, size=min(n_blk, NUM_CAMERAS), replace=False)]
        return source, preset, blackout

    def _blackout_spec(self, preset: str) -> tuple[int, bool]:
        """(고장 카메라 수, 그 채널의 검출도 지우는가).

        채널 드롭 유형은 카메라 고장이다 — 영상이 사라지면 그 카메라로 하던 검출도 사라진다.
        섞인 프리셋(heavy 등)의 블랙아웃은 기존대로 영상만 지운다.
        """
        level, typ = split_preset(preset)
        if typ == "drop":
            return DROP_CAMERAS[level], True
        if typ is not None:
            return 0, False
        return int(make_noise_config(level, risk_params=self._risk_params).n_blackout), False

    def _scene_blackout(self, scene_token: str, source: str, preset: str) -> tuple[list[int], bool]:
        """평가용 카메라 고장 — scene 내내 같은 카메라가 고장이다.

        아이템마다 고장 카메라를 다시 뽑으면, 전체 이력을 쓰는 규칙과 2초 창을 보는 모델이
        서로 다른 고장을 본다. 실제 카메라 고장도 한 주행 동안 유지된다.
        """
        if source != "noisy":
            return [], False
        k, zero_risk = self._blackout_spec(preset)
        if k <= 0:
            return [], False
        rng = np.random.default_rng([self._base_seed, self._scene_idx[scene_token], _SCENE_SALT])
        chs = [int(c) for c in rng.choice(NUM_CAMERAS, size=min(k, NUM_CAMERAS), replace=False)]
        return chs, zero_risk

    def _risk_vector(self, token: str, rng: np.random.Generator,
                     source: str, preset: str) -> np.ndarray:
        """risk_source 조건에 따른 채널 위험도 벡터(6)."""
        if source == "none":
            return np.zeros(NUM_CAMERAS, dtype=np.float32)
        contribs = self._get_contribs(token)
        if source == "gt":
            return risk_vector_from_contribs(contribs).astype(np.float32)
        if source == "noisy":
            res = apply_noise(contribs, rng, self._noise_config(preset))
            return risk_vector_from_contribs(res.contribs).astype(np.float32)
        # 향후: source == "lidar" → LiDAR 검출기 contributors 사용
        raise ValueError(f"알 수 없는 risk_source: {source}")

    def frame_risk(self, token: str, source: str | None = None,
                   preset: str | None = None) -> np.ndarray:
        """평가용 프레임 위험 벡터 — 프레임마다 고정된 손상(frame_rng).

        모델의 아이템과 규칙 기반 베이스라인이 같은 함수를 거치므로, 한 프레임에는
        방법과 무관하게 항상 같은 검출 결과가 들어간다.
        """
        source = source or self.risk_source
        preset = preset or self.noise_preset
        key = (token, source, preset)
        if key not in self._frame_risk:
            rng = frame_rng(self._base_seed, self._frame_idx[token])
            r = np.array(self._risk_vector(token, rng, source, preset), dtype=np.float32)
            chs, zero_risk = self._scene_blackout(self._frame_scene[token], source, preset)
            if zero_risk:
                r[chs] = 0.0                     # 고장 카메라로는 아무것도 검출되지 않는다
            self._frame_risk[key] = r
        return self._frame_risk[key]

    def __getitem__(self, i: int):
        scene_token, pos = self.flat[i]
        toks = self.scene_tokens[scene_token]
        # 학습 증강일 때만 epoch를 시드에 섞는다(위 '노이즈 난수 정책' 참조).
        rng = item_rng(self._base_seed, i, self._epoch if self.train_risk_noise else None)
        source, preset, blackout = self._item_noise(rng)
        if not self.train_risk_noise:
            # 평가: 카메라 고장은 scene 단위로 고정 (frame_risk와 같은 고장)
            blackout, _ = self._scene_blackout(scene_token, source, preset)

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
            feat, mask = self._get_can(tok)
            can[k, :len(feat)] = feat
            can[k, -1] = mask
            if self.train_risk_noise:
                # 학습 증강: 아이템·epoch마다 다른 실현 — 고정 손상을 암기하지 않게
                risk[k] = self._risk_vector(tok, rng, source, preset)
            else:
                # 평가: 프레임마다 하나로 고정된 검출 결과
                risk[k] = self.frame_risk(tok, source, preset)
            for ch in blackout:                     # 채널 결손: img_feat 0 (시퀀스 내내 유지)
                img[k, ch] = 0.0

        row = self.labels.loc[toks[pos]]
        soft = np.array([row[f"soft_{c}"] for c in range(NUM_CAMERAS)], dtype=np.float32)
        # 다중 선택 타깃 — 채널별 강조 여부(0/1). 진행 방향은 상시 표시라 선택 대상이
        # 아니므로 마스크로 손실에서 뺀다(타깃을 0으로만 두면 "끄는 법"을 배운다).
        aux = np.array([float(row[f"aux_{c}"]) for c in range(NUM_CAMERAS)], dtype=np.float32)
        aux_mask = np.ones(NUM_CAMERAS, dtype=np.float32)
        aux_mask[int(row["primary_view"])] = 0.0
        aux[int(row["primary_view"])] = 0.0
        return {
            "img_feat": torch.from_numpy(img),
            "can": torch.from_numpy(can),
            "risk": torch.from_numpy(risk),
            "pad_mask": torch.from_numpy(pad_mask),
            "attention_view": int(row["attention_view"]),
            "primary_view": int(row["primary_view"]),
            "layout": int(row["layout"]),
            "soft_label": torch.from_numpy(soft),
            "aux_target": torch.from_numpy(aux),
            "aux_mask": torch.from_numpy(aux_mask),
            "maneuver": str(row["maneuver"]),
            # 평가용 식별자 — 같은 프레임의 위험 판정·과거 기동 추정에 맞추고,
            # scene 시간순으로 flicker를 재기 위해 필요하다.
            "sample_token": toks[pos],
            "scene_token": scene_token,
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
    for key in ("img_feat", "can", "risk", "pad_mask", "soft_label",
                "aux_target", "aux_mask"):
        out[key] = torch.stack([b[key] for b in batch])
    for key in ("attention_view", "primary_view", "layout"):
        out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
    for key in ("maneuver", "sample_token", "scene_token"):
        out[key] = [b[key] for b in batch]
    return out
