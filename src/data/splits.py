"""scene 단위 분할 (CLAUDE.md 원칙 3 — 분할 누수 금지).

scene 단위 무작위 70/15/15 분할을 seed로 결정적으로 생성한다.
결과는 splits/scene_split_v1.json에 고정·커밋하며 이후 변경하지 않는다.
분할은 scene 이름(예: 'scene-0061') 기준 — 토큰보다 안정적·가독적.
"""

from __future__ import annotations

import json
import os


def make_scene_split(scene_names: list[str], seed: int = 42,
                     ratios=(0.70, 0.15, 0.15)) -> dict:
    """scene 이름 목록 → {train, val, test} (결정적)."""
    import numpy as np
    names = sorted(scene_names)                    # 입력 순서 무관하게 결정적
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    n = len(names)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    pick = lambda ii: sorted(names[i] for i in ii)  # noqa: E731
    return {"train": pick(idx_train), "val": pick(idx_val), "test": pick(idx_test)}


def save_split(split: dict, path: str, meta: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {"method": "scene-level random 70/15/15", **(meta or {}), **split}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def load_split(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {k: d[k] for k in ("train", "val", "test")}


def split_for_names(split: dict, available_names: set[str]) -> dict:
    """분할에서 현재 데이터셋에 존재하는 scene만 필터 (부분집합 대응)."""
    return {k: [n for n in v if n in available_names] for k, v in split.items()}
