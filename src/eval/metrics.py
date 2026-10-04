"""평가 지표 (CLAUDE.md B.8). M6에서 확장, 여기선 학습/검증 공용 최소 세트.

- top1: 예측 attention_view == 정답 비율
- topk_set: 예측이 정답 집합에 포함되면 정답 (골드셋 복수정답용)
- macro_f1: 6채널 기준 macro F1
- flicker_rate: scene 시간순 예측 전환 후 3키프레임 내 재전환 비율
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from src.utils.constants import NUM_CAMERAS


def top1(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean()) if len(pred) else 0.0


def macro_f1(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) == 0:
        return 0.0
    return float(f1_score(target, pred, labels=list(range(NUM_CAMERAS)),
                          average="macro", zero_division=0))


def multilabel_f1(pred: np.ndarray, target: np.ndarray,
                  mask: np.ndarray | None = None) -> float:
    """채널별 강조 여부(0/1)의 macro-F1 — 다중 선택 학습의 주 지표.

    단일 선택 macro_f1과 다르다: 여기서는 6채널 각각이 독립 이진 문제이고,
    0장·2장도 표현된다. mask=0인 채널(진행 방향)은 분모에서 뺀다.
    """
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(target, dtype=bool)
    m = np.ones_like(p, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if p.size == 0:
        return 0.0
    f1s = []
    for c in range(p.shape[1]):
        sel = m[:, c]
        if not sel.any():
            continue
        # 정답·예측 모두 양성이 없는 채널은 F1이 정의되지 않는다 — 평균에서 뺀다.
        # zero_division=0으로 넣으면 0점이 섞여, 집합이 100% 일치해도 F1이 0.83으로
        # 나왔다(진행 방향인 전방 채널은 보조 뷰로 켜질 일이 거의 없다).
        if not t[sel, c].any() and not p[sel, c].any():
            continue
        f1s.append(f1_score(t[sel, c], p[sel, c], zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


def mean_aux(pred: np.ndarray) -> float:
    """평균 보조 뷰 수 — 인지 부하의 대리 지표."""
    p = np.asarray(pred, dtype=bool)
    return float(p.sum(axis=1).mean()) if p.size else 0.0


def macro_ap(prob: np.ndarray, target: np.ndarray,
             mask: np.ndarray | None = None) -> float:
    """채널별 average precision의 macro 평균 — **임계값과 무관한** 순위 품질.

    모델 선택(조기 종료)에는 이 값을 쓴다. 고정 임계값 F1로 고르면, 강조 채널이
    희소해 초기에 전부 음성으로 예측하는 구간에서 F1=0이 되어 학습 신호와 선택 신호가
    어긋난다(실측: 손실은 계속 내려가는데 val F1이 0으로 붕괴). 운영 곡선 자체가
    임계값을 쓸어 그리는 것이므로, 선택도 임계값에 의존하지 않아야 한다.
    """
    p = np.asarray(prob, dtype=float)
    t = np.asarray(target, dtype=int)
    m = np.ones_like(t, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if p.size == 0:
        return 0.0
    aps = []
    for c in range(p.shape[1]):
        sel = m[:, c]
        if not sel.any() or t[sel, c].sum() == 0:
            continue                      # 양성이 없는 채널은 AP가 정의되지 않는다
        aps.append(average_precision_score(t[sel, c], p[sel, c]))
    return float(np.mean(aps)) if aps else 0.0


def threshold_for_load(prob: np.ndarray, target_load: float,
                       mask: np.ndarray | None = None,
                       aux_max: int = 2) -> float:
    """예측 평균 보조 뷰 수가 target_load에 가장 가까워지는 임계값.

    정밀도·재현율은 표시량에 좌우되므로, 비교는 **동일 표시량 조건**에서만 의미가
    있다(CLAUDE.md B.8). 라벨의 표시량에 모델을 맞춘 뒤 지표를 읽는다.
    """
    p = np.asarray(prob, dtype=float).copy()
    if mask is not None:
        p[~np.asarray(mask, dtype=bool)] = -1.0
    best, best_gap = 0.5, float("inf")
    for th in np.linspace(0.01, 0.99, 99):
        keep = p >= th
        if aux_max is not None:
            order = np.argsort(-p, axis=1)
            drop = np.zeros_like(keep)
            np.put_along_axis(drop, order[:, aux_max:], True, axis=1)
            keep = keep & ~drop
        gap = abs(keep.sum(axis=1).mean() - target_load)
        if gap < best_gap:
            best, best_gap = float(th), gap
    return best


def topk_set(pred_topk: list[list[int]], gold_sets: list[list[int]]) -> float:
    """골드셋 복수정답: 예측 top-k 중 하나라도 정답집합에 있으면 정답."""
    if not pred_topk:
        return 0.0
    hit = sum(1 for p, g in zip(pred_topk, gold_sets) if set(p) & set(g))
    return hit / len(pred_topk)


def flicker_rate(seq_by_scene: dict[str, list[int]], window: int = 3) -> float:
    """scene별 시간순 예측열 → 전환 후 window 내 재전환 비율."""
    transitions, flickers = 0, 0
    for seq in seq_by_scene.values():
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                transitions += 1
                for k in range(i + 1, min(i + 1 + window, len(seq))):
                    if seq[k] != seq[i]:
                        flickers += 1
                        break
    return flickers / transitions if transitions else 0.0
