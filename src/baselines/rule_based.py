"""규칙 기반 베이스라인 (CLAUDE.md B.7).

B1  고정 — 보조 뷰를 켜지 않는다. 진행 방향만 상시 표시하는 고정 레이아웃.
B2  규칙 — 라벨 함수의 뷰 정책을 **모델과 같은 손상 입력**에 적용, 기동은 라벨 값(오라클).
B2c 규칙 — B2와 같되 기동을 과거 궤적만으로 추정한다.

B2와 모델의 비교가 주장 C3(불완전한 입력에서 학습 모델이 규칙보다 강건하다)의 핵심이다.
공정성을 위해 세 가지를 맞춘다.

1) 같은 손상 입력. 평가에서 한 프레임의 검출 결과는 하나로 고정된다(dataset.frame_risk,
   (seed, 프레임) 시드). 모델의 아이템과 규칙이 같은 함수를 거친다.

2) 규칙을 일부러 약하게 만들지 않는다. 규칙은 scene 시작부터 지금까지의 **전체 이력**으로
   위험 EMA를 돌린다 — 실제 운용에서도 쓸 수 있는 과거 정보다. 모델의 4프레임(2초)에 맞춰
   자르면 GT 입력에서도 라벨 재현이 91%로 떨어져 "규칙을 약하게 만든 비교"가 된다.
   전체 이력이면 GT 입력에서 98.8%(F1 0.978)를 재현하고, 남는 차이는 미래 프레임이 필요한
   flicker 병합 몫이다.

3) 기동 정보. 명세(B.7)의 B2는 라벨의 기동(미래 궤적으로 판정한 오라클)을 쓴다. 모델은 과거만
   보므로 B2가 유리하다. 그래서 과거 궤적만으로 기동을 추정하는 B2c를 함께 둔다.
   C3의 주 비교 대상은 B2c이고, B2는 "기동을 미리 아는 규칙"이라는 상한 성격이다.
"""

from __future__ import annotations

import numpy as np

from src.labeling.label_function import (
    LabelConfig, infer_sample_maneuver, select_aux_views,
)
from src.utils.constants import (
    CAMERA_INDEX, NUM_CAMERAS, maneuver_view_vector, travel_direction_view,
)


class FixedFront:
    """B1 — 보조 뷰 없음. 진행 방향(primary)만 상시 표시하는 고정 레이아웃."""

    name = "B1"

    def run(self, ds) -> dict[str, dict]:
        out = {}
        for toks in ds.scene_tokens.values():
            for t in toks:
                out[t] = dict(aux=[], primary=int(ds.labels.loc[t, "primary_view"]),
                              attention=-1)
        return out


class RuleBased:
    """B2 / B2c — 라벨 함수의 뷰 정책을 손상 입력의 전체 이력에 적용한다.

    maneuver_source:
        "oracle" (B2)  라벨의 기동·진행 방향을 쓴다 — 미래 궤적으로 판정한 값.
        "causal" (B2c) 과거 궤적만으로 기동을 추정하고, 진행 방향도 거기서 정한다.
    """

    def __init__(self, cfg: LabelConfig, maneuver_source: str = "oracle", wrapper=None):
        if maneuver_source not in ("oracle", "causal"):
            raise ValueError(f"maneuver_source: {maneuver_source}")
        if maneuver_source == "causal" and wrapper is None:
            raise ValueError("causal 기동 추정에는 wrapper가 필요하다")
        self.cfg = cfg
        self.source = maneuver_source
        self.w = wrapper
        self.name = "B2" if maneuver_source == "oracle" else "B2c"
        self._causal: dict[str, str] = {}   # 임계값을 쓸며 여러 번 돌려도 한 번만 추정

    def run(self, ds) -> dict[str, dict]:
        """scene마다 시간순으로 돌며 토큰별 {aux, primary, attention}을 낸다."""
        a = self.cfg.ema_alpha
        out: dict[str, dict] = {}
        for toks in ds.scene_tokens.values():
            ema = None
            prev_primary = CAMERA_INDEX["CAM_FRONT"]
            for i, t in enumerate(toks):
                r = np.asarray(ds.frame_risk(t), dtype=np.float64)
                ema = r if ema is None else a * ema + (1 - a) * r

                if self.source == "oracle":
                    mv = str(ds.labels.loc[t, "maneuver"])
                    primary = int(ds.labels.loc[t, "primary_view"])
                else:
                    if t not in self._causal:
                        self._causal[t] = infer_sample_maneuver(self.w, t, self.cfg, causal=True)
                    mv = self._causal[t]
                    # 정지 중에는 직전 진행 방향을 유지한다(라벨의 stop_hold와 같은 취지).
                    # 극단 위험 예외는 GT TTC가 필요해 규칙 입력만으로는 판정하지 않는다.
                    if mv == "STOP" and self.cfg.stop_hold and i > 0:
                        primary = prev_primary
                    else:
                        primary = travel_direction_view(mv)
                prev_primary = primary

                m_vec = np.asarray(maneuver_view_vector(mv), dtype=np.float64)
                aux = select_aux_views(ema, m_vec, primary, self.cfg)
                s = ema + self.cfg.lambda_m * m_vec
                s[primary] = -np.inf
                out[t] = dict(aux=[int(c) for c in aux], primary=primary,
                              attention=int(np.argmax(s)))
        return out


def as_arrays(pred: dict[str, dict], tokens: list[str]):
    """토큰 순서에 맞춘 (선택 (N,6) bool, primary (N,), attention (N,))."""
    sel = np.zeros((len(tokens), NUM_CAMERAS), dtype=bool)
    prim = np.zeros(len(tokens), dtype=int)
    att = np.zeros(len(tokens), dtype=int)
    for i, t in enumerate(tokens):
        p = pred[t]
        sel[i, p["aux"]] = True
        prim[i] = p["primary"]
        att[i] = p["attention"]
    return sel, prim, att
