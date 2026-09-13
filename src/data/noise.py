"""검출 오류 시뮬레이터 (CLAUDE.md B.6, 원칙 2의 구현체).

실전에는 GT가 없고 객체 탐지기의 불완전한 결과만 있다. 이를 흉내내기 위해
GT 기반 위험 기여(contributors)에 오류를 주입한 뒤 위험도 벡터를 재계산한다.

오류 유형:
  - 누락(miss): 각 객체를 확률 p_miss로 제거 (원거리일수록 ↑, p=p_miss·min(1,r/30)).
  - 오검출(FP): 프레임당 λ_fp ~ Poisson 개의 가짜 객체를 무작위 채널·거리에 추가.
  - 위치/속도 노이즈: 거리 r·TTC에 가우시안 잡음 → w 재계산.
  - 채널 블랙아웃: 지정 수의 채널 img_feat을 0으로(위험도 유지/제거 옵션).

강도 프리셋: clean / light / medium / heavy (M7 강건성 x축).
학습 증강(train.risk_noise)과 평가(강건성) 양쪽에서 쓰인다.

향후 확장(risk_source: lidar): LiDAR/융합 검출기 결과를 contributors로 만들어
같은 파이프라인에 넣으면, 노이즈 강도 대신 '실전 검출 품질' 축으로 평가 가능.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.labeling.ttc import risk_weight
from src.utils.constants import NUM_CAMERAS


@dataclass
class NoiseConfig:
    p_miss: float = 0.0          # 기본 누락 확률
    lambda_fp: float = 0.0       # 프레임당 평균 FP 개수
    sigma_pos: float = 0.0       # 위치 노이즈 표준편차(m), 거리 비례 가산
    sigma_ttc_rel: float = 0.0   # TTC 상대 노이즈(속도 오차 대리)
    n_blackout: int = 0          # 블랙아웃 채널 수
    blackout_risk_zero: bool = False  # 블랙아웃 채널의 위험도도 0으로 할지
    # w 재계산용 위험 파라미터 (LabelConfig와 정합)
    tau: float = 4.0
    w_type: str = "exp"
    w_min_near: float = 0.3
    near_range: float = 10.0


PRESETS = {
    "clean":  dict(p_miss=0.0,  lambda_fp=0.0, sigma_pos=0.0, sigma_ttc_rel=0.0, n_blackout=0),
    "light":  dict(p_miss=0.10, lambda_fp=0.3, sigma_pos=0.5, sigma_ttc_rel=0.15, n_blackout=0),
    "medium": dict(p_miss=0.25, lambda_fp=1.0, sigma_pos=1.5, sigma_ttc_rel=0.30, n_blackout=0),
    "heavy":  dict(p_miss=0.45, lambda_fp=2.0, sigma_pos=3.0, sigma_ttc_rel=0.50, n_blackout=1),
}


def make_noise_config(preset: str = "clean", risk_params: dict | None = None,
                      **overrides) -> NoiseConfig:
    """프리셋 + 위험 파라미터로 NoiseConfig 생성."""
    params = dict(PRESETS[preset])
    params.update(risk_params or {})
    params.update(overrides)
    known = {k: v for k, v in params.items() if k in NoiseConfig.__dataclass_fields__}
    return NoiseConfig(**known)


@dataclass
class NoiseResult:
    contribs: list[dict]
    blackout: list[int]          # img_feat을 0으로 만들 채널 인덱스


def apply_noise(contribs: list[dict], rng: np.random.Generator,
                nc: NoiseConfig) -> NoiseResult:
    """GT 기여 목록 → 오류 주입된 기여 목록 + 블랙아웃 채널."""
    out: list[dict] = []

    # 1) 누락 + 위치/속도 노이즈
    for c in contribs:
        r = c["r"]
        p_drop = nc.p_miss * min(1.0, r / 30.0)
        if rng.random() < p_drop:
            continue
        r2 = max(0.0, r + rng.normal(0.0, nc.sigma_pos * (1.0 + r / 30.0))) \
            if nc.sigma_pos > 0 else r
        ttc = c["ttc"]
        if np.isfinite(ttc) and nc.sigma_ttc_rel > 0:
            ttc = max(0.1, ttc * (1.0 + rng.normal(0.0, nc.sigma_ttc_rel)))
        # ρ(상태·경로 계수)는 검출기가 따로 추정하는 양이므로 위치 노이즈로 바꾸지 않는다.
        # 속성 오분류는 향후 별도 노이즈 유형으로 추가한다.
        rho = float(c.get("rho", 1.0))
        w = risk_weight(ttc, r2, tau=nc.tau, w_type=nc.w_type,
                        w_min_near=nc.w_min_near, near_range=nc.near_range, rho=rho)
        out.append({"vis": list(c["vis"]), "r": r2, "ttc": ttc, "w": w, "rho": rho})

    # 2) 오검출(FP)
    if nc.lambda_fp > 0:
        for _ in range(int(rng.poisson(nc.lambda_fp))):
            ch = int(rng.integers(NUM_CAMERAS))
            r = float(rng.uniform(5.0, 40.0))
            ttc = float(rng.uniform(1.0, 10.0))
            w = risk_weight(ttc, r, tau=nc.tau, w_type=nc.w_type,
                            w_min_near=nc.w_min_near, near_range=nc.near_range)
            # 오검출은 '움직이는 실제 객체'로 보이므로 ρ=1 — 가장 교란적인 형태
            out.append({"vis": [ch], "r": r, "ttc": ttc, "w": w, "rho": 1.0})

    # 3) 채널 블랙아웃
    blackout: list[int] = []
    if nc.n_blackout > 0:
        blackout = list(rng.choice(NUM_CAMERAS, size=min(nc.n_blackout, NUM_CAMERAS),
                                   replace=False))
        if nc.blackout_risk_zero:
            out = [c for c in out if not any(ci in blackout for ci in c["vis"])]

    return NoiseResult(contribs=out, blackout=blackout)
