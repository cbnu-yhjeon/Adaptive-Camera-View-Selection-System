"""위험 기여(contributors) + CAN 특징 디스크 캐시 (M3 보강).

Dataset은 keyframe마다 GT 박스를 6개 카메라에 투영해 위험 기여를 구해야 하는데,
trainval 34,149 keyframe에서 이 계산은 실행마다 수십 분이 든다. 실험 설계상
학습·평가를 최소 10회 이상 반복하므로(베이스라인 4 + ablation 5 + 강건성 평가),
같은 입력에서 매번 다시 계산할 이유가 없다. 라벨 생성과 동일한 결정적 입력이므로
한 번 계산해 npz로 저장하고 이후 재사용한다.

캐시 키는 **contributors 값을 바꾸는 LabelConfig 필드만** 반영한다. 예를 들어
tau를 바꾸면 w가 달라지므로 새 캐시가 되지만, ema_alpha·T_soft처럼 라벨 결합
단계에만 쓰이는 값은 캐시를 무효화하지 않는다.

저장 형식 (ragged, npz)
    tokens   (n,)    <U32    keyframe 토큰 (순서 = 행 인덱스)
    offsets  (n+1,)  int64   keyframe i의 객체 구간 [offsets[i], offsets[i+1])
    vis_mask (m,)    uint8   객체가 보이는 채널 비트마스크 (bit c → 채널 c)
    obj_r    (m,)    float32 수평 거리
    obj_ttc  (m,)    float32 TTC (inf 포함)
    obj_w    (m,)    float32 위험 가중 (ρ 반영 후)
    obj_rho  (m,)    float32 상태·경로 계수 ρ — 노이즈 재계산 시 그대로 재사용
    obj_dca  (m,)    float32 최접근 거리 — '임박 위험' 판정과 평가 지표에 사용
    obj_floor(m,)    float32 근접 보정 실효 하한 (진행 방향 조건 반영)
    can_feat (n,F)   float32 정규화된 CAN 특징 (F = len(CAN_FEATURES))
    can_mask (n,)    float32 1=CAN 존재
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from src.labeling.label_function import LabelConfig, channel_risk_contributors
from src.utils.constants import CAN_FEATURES, NUM_CAMERAS

# contributors 값에 실제로 영향을 주는 필드만 캐시 키에 넣는다.
_KEY_FIELDS = ("categories", "max_range", "tau", "w_type",
               "w_min_near", "near_range", "v_r_min",
               "ttc_mode", "d_near", "d_scale", "rho_min", "attr_weight",
               "attr_exempt_touch", "attr_exempt_scale", "touch_range")


def cache_key(cfg: LabelConfig) -> str:
    # CAN 특징 구성도 키에 넣는다. 없으면 특징을 늘려도 옛 캐시가 그대로 재사용돼
    # 새 특징이 에러 없이 0으로 들어간다(지시등 추가 때 실제로 그럴 뻔했다).
    payload = json.dumps({**{k: getattr(cfg, k) for k in _KEY_FIELDS},
                          "can_features": CAN_FEATURES},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def cache_path(cache_dir: str, cfg: LabelConfig) -> str:
    return os.path.join(cache_dir, f"contribs_{cache_key(cfg)}.npz")


def build(wrapper, tokens: list[str], cfg: LabelConfig, path: str,
          verbose: bool = True) -> str:
    """contributors + CAN 특징을 계산해 npz로 저장."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(tokens)
    offsets = np.zeros(n + 1, dtype=np.int64)
    vis_l: list[int] = []
    r_l: list[float] = []
    ttc_l: list[float] = []
    w_l: list[float] = []
    rho_l: list[float] = []
    dca_l: list[float] = []
    floor_l: list[float] = []
    can_feat = np.zeros((n, len(CAN_FEATURES)), dtype=np.float32)
    can_mask = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for i, tok in enumerate(tokens):
        for c in channel_risk_contributors(wrapper, tok, cfg):
            mask = 0
            for ci in c["vis"]:
                mask |= 1 << ci
            vis_l.append(mask)
            r_l.append(c["r"])
            ttc_l.append(c["ttc"])
            w_l.append(c["w"])
            rho_l.append(c.get("rho", 1.0))
            dca_l.append(c.get("dca", 0.0))
            floor_l.append(c.get("floor", 0.0))
        offsets[i + 1] = len(vis_l)

        feat, m = wrapper.get_can_features(tok)
        can_feat[i, :len(feat)] = feat
        can_mask[i] = m

        if verbose and (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  [기여캐시] {i+1}/{n}  {el:.0f}s  "
                  f"(남은 예상 {el/(i+1)*(n-i-1):.0f}s)")

    np.savez(path,
             tokens=np.array(tokens),
             offsets=offsets,
             vis_mask=np.array(vis_l, dtype=np.uint8),
             obj_r=np.array(r_l, dtype=np.float32),
             obj_ttc=np.array(ttc_l, dtype=np.float32),
             obj_w=np.array(w_l, dtype=np.float32),
             obj_rho=np.array(rho_l, dtype=np.float32),
             obj_dca=np.array(dca_l, dtype=np.float32),
             obj_floor=np.array(floor_l, dtype=np.float32),
             can_feat=can_feat,
             can_mask=can_mask)
    if verbose:
        print(f"[기여캐시] {n} keyframe / 객체 {len(vis_l)}개, "
              f"{time.time()-t0:.0f}s → {path}")
    return path


class ContribCache:
    """캐시 로더. get(token) → contributors 목록, can(token) → (feat[5], mask)."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.tokens = [str(t) for t in z["tokens"]]
        self.row = {t: i for i, t in enumerate(self.tokens)}
        self.offsets = z["offsets"]
        self.vis_mask = z["vis_mask"]
        self.obj_r = z["obj_r"]
        self.obj_ttc = z["obj_ttc"]
        self.obj_w = z["obj_w"]
        self.obj_rho = z["obj_rho"] if "obj_rho" in z else np.ones_like(z["obj_w"])
        self.obj_dca = z["obj_dca"] if "obj_dca" in z else np.zeros_like(z["obj_w"])
        self.obj_floor = z["obj_floor"] if "obj_floor" in z else np.zeros_like(z["obj_w"])
        self.can_feat = z["can_feat"]
        if self.can_feat.shape[1] != len(CAN_FEATURES):
            raise ValueError(
                f"기여 캐시의 CAN 특징 폭 {self.can_feat.shape[1]} ≠ 현재 구성 "
                f"{len(CAN_FEATURES)} ({path}) — 캐시를 다시 만들어야 한다")
        self.can_mask = z["can_mask"]

    def __contains__(self, token: str) -> bool:
        return token in self.row

    def __len__(self) -> int:
        return len(self.tokens)

    def get(self, token: str) -> list[dict]:
        """noise.apply_noise / risk_vector_from_contribs가 기대하는 형식 그대로 반환."""
        i = self.row[token]
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        out = []
        for j in range(a, b):
            m = int(self.vis_mask[j])
            vis = [c for c in range(NUM_CAMERAS) if m & (1 << c)]
            out.append({"vis": vis, "r": float(self.obj_r[j]),
                        "ttc": float(self.obj_ttc[j]), "w": float(self.obj_w[j]),
                        "rho": float(self.obj_rho[j]),
                        "dca": float(self.obj_dca[j]),
                        "floor": float(self.obj_floor[j])})
        return out

    def can(self, token: str) -> tuple[np.ndarray, float]:
        i = self.row[token]
        return self.can_feat[i], float(self.can_mask[i])


def load_or_build(wrapper, tokens: list[str], cfg: LabelConfig, cache_dir: str,
                  force: bool = False, verbose: bool = True) -> ContribCache:
    """캐시가 있으면 로드, 없으면 생성 후 로드.

    캐시에 없는 토큰이 있으면(부분 데이터로 만든 캐시를 더 큰 데이터에 쓰는 경우)
    재생성한다.
    """
    path = cache_path(cache_dir, cfg)
    if os.path.isfile(path) and not force:
        cache = ContribCache(path)
        missing = [t for t in tokens if t not in cache]
        if not missing:
            if verbose:
                print(f"[기여캐시] 재사용 ({len(cache)} keyframe) ← {path}")
            return cache
        if verbose:
            print(f"[기여캐시] {len(missing)}개 keyframe 누락 → 재생성")
    build(wrapper, tokens, cfg, path, verbose=verbose)
    return ContribCache(path)
