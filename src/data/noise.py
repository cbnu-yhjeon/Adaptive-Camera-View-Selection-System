"""검출 오류 시뮬레이터 (CLAUDE.md B.6, 원칙 2의 구현체).

실전에는 GT가 없고 객체 탐지기의 불완전한 결과만 있다. 이를 흉내내기 위해
GT 기반 위험 기여(contributors)에 오류를 주입한 뒤 위험도 벡터를 재계산한다.

오류 유형:
  - 누락(miss): 각 객체를 확률 p_miss로 제거 (원거리일수록 ↑, p=p_miss·min(1,r/30)).
  - 오검출(FP): 프레임당 λ_fp ~ Poisson 개의 가짜 객체를 무작위 채널(화각 안)·거리에 추가.
    가짜 객체는 **정지된 잡음 + 추적 속도 오차**다(fp_mode="static", 2026-10-07). 위험도는 실제
    객체와 같은 계산(최접근 TTC·DCA·경로 계수·근접 보정)으로 내 차의 움직임에서 나온다 —
    내 차가 서 있으면 가짜 객체는 (속도 오차만큼만) 다가온다.
    이전 방식(fp_mode="approaching")은 차량 상태와 무관하게 TTC 1~10초를 줬다. 그러면 정지 중
    θ를 넘는 위험 채널의 62%(medium)가 가짜가 되어(주행 중 41%), 모델이 '정지 중 위험도는
    가짜'라고 배웠다 — 정지 프레임 포착률이 규칙보다 10%p 낮았던 원인. 실제 검출기의 오검출은
    대개 정지된 잡음(반사·배경 오인)이라 내 차가 서 있으면 다가오지 않는다.
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

from src.labeling.ttc import closest_approach, compute_ttc, near_floor, path_factor, risk_weight
from src.utils.constants import CAMERA_HFOV_DEG, CAMERA_YAW_DEG, CAMERAS, NUM_CAMERAS


@dataclass
class NoiseConfig:
    p_miss: float = 0.0          # 기본 누락 확률
    lambda_fp: float = 0.0       # 프레임당 평균 FP 개수
    sigma_pos: float = 0.0       # 위치 노이즈 표준편차(m), 거리 비례 가산
    sigma_ttc_rel: float = 0.0   # TTC 상대 노이즈(속도 오차 대리)
    n_blackout: int = 0          # 블랙아웃 채널 수
    blackout_risk_zero: bool = False  # 블랙아웃 채널의 위험도도 0으로 할지
    sigma_fp_vel: float = 0.0    # 오검출 객체의 속도 추정 오차(m/s, 축별 가우시안)
    fp_mode: str = "static"      # static: 정지 잡음 | approaching: 이전 방식(결과 재현용)
    # w 재계산용 위험 파라미터 (LabelConfig와 정합)
    tau: float = 4.0
    w_type: str = "exp"
    w_min_near: float = 0.3
    near_range: float = 10.0
    v_r_min: float = 0.1
    ttc_mode: str = "cpa"
    d_near: float = 4.0
    d_scale: float = 3.0
    rho_min: float = 0.05
    touch_range: float = 2.0
    stop_speed: float = 0.5


PRESETS = {
    "clean":  dict(p_miss=0.0,  lambda_fp=0.0, sigma_pos=0.0, sigma_ttc_rel=0.0, n_blackout=0),
    "light":  dict(p_miss=0.10, lambda_fp=0.3, sigma_pos=0.5, sigma_ttc_rel=0.15, n_blackout=0, sigma_fp_vel=0.5),
    "medium": dict(p_miss=0.25, lambda_fp=1.0, sigma_pos=1.5, sigma_ttc_rel=0.30, n_blackout=0, sigma_fp_vel=1.0),
    "heavy":  dict(p_miss=0.45, lambda_fp=2.0, sigma_pos=3.0, sigma_ttc_rel=0.50, n_blackout=1, sigma_fp_vel=2.0),
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
                nc: NoiseConfig, ego_speed: float = 0.0) -> NoiseResult:
    """GT 기여 목록 → 오류 주입된 기여 목록 + 블랙아웃 채널.

    ego_speed: 내 차의 전진 속도(m/s, 후진은 음수) — 오검출 객체의 위험 계산에 쓴다.
    """
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
        # 근접 보정 하한은 진행 방향에 따라 라벨 단계에서 결정된 값을 그대로 쓴다.
        floor = float(c.get("floor", nc.w_min_near))
        w = risk_weight(ttc, r2, tau=nc.tau, w_type=nc.w_type,
                        w_min_near=floor, near_range=nc.near_range, rho=rho)
        out.append({"vis": list(c["vis"]), "r": r2, "ttc": ttc, "w": w,
                    "rho": rho, "floor": floor})

    # 2) 오검출(FP)
    if nc.lambda_fp > 0:
        for _ in range(int(rng.poisson(nc.lambda_fp))):
            ch = int(rng.integers(NUM_CAMERAS))
            r = float(rng.uniform(5.0, 40.0))
            if nc.fp_mode == "approaching":            # 이전 방식 — 2026-10-07 이전 결과 재현용
                ttc = float(rng.uniform(1.0, 10.0))
                w = risk_weight(ttc, r, tau=nc.tau, w_type=nc.w_type,
                                w_min_near=nc.w_min_near, near_range=nc.near_range)
                out.append({"vis": [ch], "r": r, "ttc": ttc, "w": w, "rho": 1.0, "fp": True})
                continue
            out.append(_static_phantom(ch, r, rng, nc, ego_speed))

    # 3) 채널 블랙아웃
    blackout: list[int] = []
    if nc.n_blackout > 0:
        blackout = list(rng.choice(NUM_CAMERAS, size=min(nc.n_blackout, NUM_CAMERAS),
                                   replace=False))
        if nc.blackout_risk_zero:
            out = [c for c in out if not any(ci in blackout for ci in c["vis"])]

    return NoiseResult(contribs=out, blackout=blackout)


def _static_phantom(ch: int, r: float, rng: np.random.Generator, nc: NoiseConfig,
                    ego_speed: float) -> dict:
    """정지 잡음 오검출 하나 — 실제 객체와 같은 위험 계산을 차량 좌표계에서 한다.

    위치는 그 카메라 화각 안의 무작위 방위, 속도는 추적 오차(σ_fp_vel)뿐이다. 상태 속성은
    검출기가 알 수 없으므로 '움직이는 객체'(ρ_attr=1)로 본다 — 가장 교란적인 쪽.
    """
    cam = CAMERAS[ch]
    half = CAMERA_HFOV_DEG[cam] / 2.0
    th = np.radians(CAMERA_YAW_DEG[cam] + rng.uniform(-half, half))
    p = np.array([r * np.cos(th), r * np.sin(th)])
    v_obj = rng.normal(0.0, nc.sigma_fp_vel, 2) if nc.sigma_fp_vel > 0 else np.zeros(2)
    v_ego = np.array([float(ego_speed), 0.0])
    o = np.zeros(2)
    ttc = compute_ttc(p, o, v_obj, v_ego, v_r_min=nc.v_r_min, mode=nc.ttc_mode)
    _, dca = closest_approach(p, o, v_obj, v_ego)
    rho = path_factor(dca, d_near=nc.d_near, d_scale=nc.d_scale, rho_min=nc.rho_min)
    floor = near_floor(p, o, v_ego, r, w_min_near=nc.w_min_near, near_range=nc.near_range,
                       touch_range=nc.touch_range, stop_speed=nc.stop_speed)
    w = risk_weight(ttc, r, tau=nc.tau, w_type=nc.w_type, w_min_near=floor,
                    near_range=nc.near_range, rho=rho)
    return {"vis": [ch], "r": r, "ttc": float(ttc), "w": float(w), "rho": float(rho),
            "dca": float(dca), "floor": float(floor), "fp": True}

