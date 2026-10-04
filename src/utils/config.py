"""설정 로딩 — base / label / train 세 파일과 train 설정의 상속.

ablation·베이스라인 설정이 제안 모델 설정을 통째로 복사하면, 제안 모델 쪽을 고쳤을 때
복사본이 어긋나 "제거 효과"가 아니라 "설정 차이"를 재게 된다. 그래서 train 설정은
`inherit:`로 다른 설정을 물려받고 **바뀌는 값만** 적는다.

    # configs/ablation_a2_no_risk.yaml
    inherit: train_proposed.yaml
    exp_name: ablation_a2_no_risk
    model:
      use_risk: false
"""

from __future__ import annotations

import copy
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_train_config(path: str, _seen: tuple = ()) -> dict:
    """train 설정을 읽고 `inherit:` 체인을 따라 병합한다(자식 값이 이긴다)."""
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"설정 상속이 순환한다: {' → '.join(_seen + (path,))}")
    cfg = load_yaml(path)
    parent = cfg.pop("inherit", None)
    if parent is None:
        return cfg
    parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
    return _deep_merge(load_train_config(parent_path, _seen + (path,)), cfg)


def load_cfgs(train_cfg_path: str) -> tuple[dict, dict, dict]:
    """(base, label, train) — 모든 스크립트가 같은 방식으로 읽는다."""
    base = load_yaml(os.path.join(CONFIG_DIR, "base.yaml"))
    lab = load_yaml(os.path.join(CONFIG_DIR, "label_v1.yaml"))
    return base, lab, load_train_config(train_cfg_path)
