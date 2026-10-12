"""그림 2 — 상황별 정성 사례 (CLAUDE.md M8).

TURN · LANE_CHANGE · REVERSE 상황마다 성공 2 + 실패 1 사례를 고르고, 6채널 이미지 위에
제안 모델 · B2(규칙, 오라클 기동) · 골드 정답이 각각 어느 화면을 골랐는지 나란히 표시한다.

  성공/실패  제안 모델의 1순위 보조 화면(Top-1)이 골드 정답 집합에 드는가. 성공 사례는
            같은 정보 조건의 규칙(B2c)이 틀린 장면을 먼저 고른다 — 차이가 드러나는 장면.
  상황      의사 라벨의 기동으로 나눈다(판정 근거가 아니라 묶음 기준).
  입력      표 1의 공정 비교 조건 noisy(medium). 모델 임계값은 val에서 라벨 표시량에 맞춘다.

골드셋에 그 상황이 없으면 test 분할 전체에서 고르고 정답 칸에 의사 라벨을 대신 표시한다
(그림에 '골드 없음 — 의사 라벨 기준'으로 적는다). 평가 분할에 아예 없는 상황은 그리지 않고
cases.json에 이유를 남긴다 — 후진은 trainval 전체 8프레임(scene-0731·0945)이 모두 학습 분할이라
학습한 장면을 결과로 보여주게 된다.
선택은 토큰 순서로 결정적이다(재실행해도 같은 사례).

출력: results/fig2/fig2_{상황}.png · results/fig2/cases.json

사용
    python scripts/08_make_figures.py --gold data/goldset_v1.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.rule_based import RuleBased  # noqa: E402
from src.eval.metrics import threshold_for_load  # noqa: E402
from src.models.proposed import ModelConfig, ViewSelector  # noqa: E402
from src.utils.config import load_cfgs  # noqa: E402
from src.utils.constants import CAMERAS  # noqa: E402
from src.utils.plotting import use_korean_font  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("ev", "06_evaluate.py")
gs = _load("gs", "06b_goldset_sample.py")

GRID = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
KO = {"CAM_FRONT_LEFT": "좌전측방", "CAM_FRONT": "전방", "CAM_FRONT_RIGHT": "우전측방",
      "CAM_BACK_LEFT": "좌후측방", "CAM_BACK": "후방", "CAM_BACK_RIGHT": "우후측방"}
SITUATIONS = {"TURN": ("TURN_LEFT", "TURN_RIGHT"),
              "LANE_CHANGE": ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"),
              "REVERSE": ("REVERSE",)}
COL = {"gold": "#d4a017", "model": "#1f5fa8", "b2": "#7a8288", "primary": "#0d6b73"}


def predictions(env, split, device):
    """제안 모델·B2·B2c의 선택과 순위 — 표 1과 같은 noisy(medium) 입력, 같은 운영점."""
    _, _, tc = load_cfgs("configs/train_proposed.yaml")
    ckpt = ev.latest_ckpt(tc.get("exp_name", "proposed"), env.base["version"])
    st = torch.load(ckpt, map_location=device, weights_only=False)
    model = ViewSelector(ModelConfig(**st["mcfg"])).to(device)
    model.load_state_dict(st["model"])
    model.eval()
    Pv, _, mv = ev.run_model(model, env.dataset("val", "noisy", "medium"), device)
    th = threshold_for_load(Pv, float((mv["aux"] > 0.5).sum(1).mean()), mv["mask"] > 0.5, env.cfg.aux_max)
    P, _, meta = ev.run_model(model, env.dataset(split, "noisy", "medium"), device)
    prim = meta["primary"].astype(int)
    sel = ev.select_from_probs(P, prim, th, env.cfg.aux_max)
    out = {}
    for i, t in enumerate(meta["token"]):
        r = P[i].copy(); r[prim[i]] = -1
        out[t] = dict(primary=int(prim[i]), model_sel=np.flatnonzero(sel[i]).tolist(), model_top1=int(r.argmax()))
    ds = env.dataset(split, "noisy", "medium")
    for name, src in (("b2", "oracle"), ("b2c", "causal")):
        rule = RuleBased(copy.deepcopy(env.cfg), src, wrapper=env.w)
        for t, p in rule.run(ds).items():
            if t in out:
                out[t][f"{name}_sel"] = list(p["aux"])
                out[t][f"{name}_top1"] = int(p["attention"])
    return out, ckpt, float(th)


def pick_cases(env, pred, gold, split):
    """상황별 성공 2 + 실패 1. 골드가 없는 상황은 의사 라벨을 정답 대신 쓴다."""
    L = env.df.set_index("sample_token")
    cases = {}
    for sit, mvs in SITUATIONS.items():
        g_toks = sorted(t for t in (gold or {}) if t in pred and L.loc[t, "maneuver"] in mvs and gold[t]["set"])
        source = "gold"
        if not g_toks:                                  # 골드셋에 그 상황이 없다 → 의사 라벨 기준
            source = "pseudo"
            g_toks = sorted(t for t in pred if L.loc[t, "maneuver"] in mvs)
        def answer(t):
            if source == "gold":
                return sorted(gold[t]["set"] - {pred[t]["primary"]})
            return [c for c in range(len(CAMERAS)) if L.loc[t, f"aux_{c}"] > 0.5 and c != pred[t]["primary"]]
        ok = [t for t in g_toks if answer(t) and pred[t]["model_top1"] in answer(t)]
        bad = [t for t in g_toks if answer(t) and pred[t]["model_top1"] not in answer(t)]
        # 성공 사례는 B2c가 틀린 장면을 먼저 — 같은 정보 조건의 규칙과 차이가 나는 장면
        ok.sort(key=lambda t: (pred[t].get("b2c_top1") in answer(t), t))
        chosen, used = [], set()                         # 사례마다 다른 scene
        for kind, pool, k in (("성공", ok, 2), ("실패", bad, 1)):
            for t in pool:
                sc = env.w.sample_scene_token(t)
                if sc in used or sum(c[0] == kind for c in chosen) >= k:
                    continue
                chosen.append((kind, t)); used.add(sc)
        note = None
        if not g_toks:
            n_all = int(L["maneuver"].isin(mvs).sum())
            note = f"평가 분할({split})에 이 상황이 없다 — trainval 전체 {n_all}프레임, 모두 다른 분할"
        cases[sit] = dict(source=source, n_pool=len(g_toks), note=note, items=[
            dict(kind=k, token=t, answer=answer(t), maneuver=str(L.loc[t, "maneuver"]), **pred[t]) for k, t in chosen])
    return cases


def draw(env, sit, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    use_korean_font()
    rows = info["items"]
    if not rows:
        return False
    fig, axes = plt.subplots(len(rows), 6, figsize=(16, 2.55 * len(rows) + 0.6), squeeze=False)
    src = "골드 정답" if info["source"] == "gold" else "의사 라벨(골드 없음)"
    for r, it in enumerate(rows):
        t = it["token"]
        paths = env.w.get_image_paths(t)
        sc = env.w.sample_scene_token(t)
        vm = env.w.get_can_messages(sc, "vehicle_monitor")
        v, sig = gs.frame_state(env.w, t, vm)
        for c, cam in enumerate(GRID):
            ax = axes[r, c]
            ci = CAMERAS.index(cam)
            im = Image.open(paths[cam]).convert("RGB")
            im.thumbnail((480, 270))
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            marks = []
            if ci == it["primary"]:
                marks.append(("진행", COL["primary"]))
            if ci in it["answer"]:
                marks.append(("정답", COL["gold"]))
            if ci in it["model_sel"] or ci == it["model_top1"]:
                marks.append(("제안" + ("①" if ci == it["model_top1"] else ""), COL["model"]))
            if ci in it.get("b2_sel", []):
                marks.append(("B2", COL["b2"]))
            edge = COL["gold"] if ci in it["answer"] else (COL["primary"] if ci == it["primary"] else "#ffffff")
            for sp in ax.spines.values():
                sp.set_edgecolor(edge); sp.set_linewidth(4 if edge != "#ffffff" else 0.5)
            for k, (label, col) in enumerate(marks):
                ax.text(0.03 + 0.21 * k, 0.93, label, transform=ax.transAxes, va="top",
                        color="white", fontsize=8, fontweight="bold",
                        bbox=dict(facecolor=col, edgecolor="none", pad=1.5))
            if r == 0:
                ax.set_title(KO[cam], fontsize=10)
        gold_ko = ", ".join(KO[CAMERAS[c]] for c in it["answer"]) or "강조 불필요"
        model_ko = KO[CAMERAS[it["model_top1"]]]
        axes[r, 0].set_ylabel(f"{it['kind']}\n{env.w.scene_name(sc)}", fontsize=10)
        axes[r, 3].set_xlabel(
            f"{it['maneuver']} · {abs(v):.0f} km/h · 지시등 {sig or '정보 없음'}  |  {src}: {gold_ko}  ·  제안 1순위: {model_ko}"
            + (f"  ·  B2c 1순위: {KO[CAMERAS[it['b2c_top1']]]}" if "b2c_top1" in it else ""), fontsize=9)
    fig.suptitle(f"그림 2 — {sit} 상황 (입력 noisy-medium · 노란 테두리 = {src}, 청록 = 진행 방향)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main() -> int:
    base, lab, tc = load_cfgs("configs/train_proposed.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/goldset_v1.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--version", default=None, help="mini 스모크: v1.0-mini")
    ap.add_argument("--splits", default=None, help="mini 스모크: splits/scene_split_mini.json")
    ap.add_argument("--out_dir", default="results/fig2")
    args = ap.parse_args()

    version = args.version or base["version"]
    env = ev.Env(base, lab, tc, version, args.splits, gold_path=args.gold)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred, ckpt, th = predictions(env, args.split, device)
    cases = pick_cases(env, pred, env.gold, args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    for sit, info in cases.items():
        ok = draw(env, sit, info, os.path.join(args.out_dir, f"fig2_{sit}.png"))
        print(f"  {sit:12s} 기준 {info['source']:6s} 후보 {info['n_pool']:4d}  사례 "
              + ", ".join(f"{x['kind']}" for x in info["items"]) + ("" if ok else f"  (그리지 않음: {info['note']})"))
    with open(os.path.join(args.out_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(dict(split=args.split, condition="noisy-medium", ckpt=ckpt, threshold=th,
                       gold=args.gold if env.gold else None, n_gold=len(env.gold or {}), cases=cases),
                  f, ensure_ascii=False, indent=2)
    print(f"✅ {args.out_dir}/fig2_*.png · cases.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
