"""강건성 열화 곡선 (CLAUDE.md B.9, M7) — 주장 C3의 실험.

노이즈를 **유형별**로 나눠 강도를 올리며, 같은 손상 입력에서 규칙(B2·B2c)과 모델이 얼마나
무너지는지 잰다. 유형을 섞은 프리셋 하나로는 "무엇에 강하고 무엇에 약한가"를 말할 수 없다.

  유형  miss  누락        각 객체를 p_miss로 제거 (원거리일수록 ↑)
        fp    오검출      프레임당 Poisson(λ_fp)개의 가짜 객체
        pos   위치 오차   거리·TTC 가우시안 잡음
        drop  채널 드롭   카메라 k대 고장 — 그 채널의 영상 특징과 검출이 모두 사라진다.
                         고장은 scene 내내 유지된다(규칙의 전체 이력과 모델이 같은 고장을 본다).
  강도  clean · light · medium · heavy   (drop: 0 · 1 · 2 · 3대)

지표는 **같은 표시량에서의 위험 사건 포착률**이다(B.8). 고정 운영점으로 비교하면 오검출 때문에
보조 뷰를 많이 켠 규칙이 포착률에서 이긴다. 각 방법의 운영 곡선(규칙 θ, 모델 τ)을 그리고
기준 표시량에서 선형 보간한다. 기준 표시량은 해당 분할 라벨의 평균 보조 뷰 수다.
골드셋 Top-1(명세 B.9의 y축)은 골드셋 라벨링이 끝나면 추가한다.

산출
  results/robustness/robustness_{split}.json   모든 조건의 곡선과 보간값
  results/robustness/robustness_{split}.md     표
  results/robustness/robustness_{split}.png    유형별 4개 서브플롯

사용
    python scripts/07_robustness.py --split val
    python scripts/07_robustness.py --split test --extra_ckpt results/ablation_a2_no_risk/<run>/best.ckpt
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
from src.models.proposed import ModelConfig, ViewSelector  # noqa: E402
from src.utils.config import load_cfgs  # noqa: E402
from src.utils.plotting import use_korean_font  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ev", os.path.join(os.path.dirname(__file__), "06_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

TYPES = ["miss", "fp", "pos", "drop"]
LEVELS = ["clean", "light", "medium", "heavy"]
TYPE_KO = {"miss": "누락", "fp": "오검출", "pos": "위치 오차", "drop": "채널 드롭"}


def interp(pts, key, load):
    xs = np.array([p["mean_aux"] for p in pts]); ys = np.array([p[key] for p in pts])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if load < xs[0] or load > xs[-1]:
        return None
    return float(np.interp(load, xs, ys))


def rule_curve(env, rule, ds):
    cfg = rule.cfg
    keep = cfg.aux_threshold
    pts = []
    for th in ev.CURVE_TH:
        cfg.aux_threshold = th
        sel, att, meta = ev.run_rule(rule, ds)
        r = ev.metrics(env, sel, att, meta)
        pts.append(dict(param=th, **{k: r[k] for k in ("mean_aux", "recall", "precision", "flicker")}))
    cfg.aux_threshold = keep
    return pts


def model_curve(env, model, ds, device):
    P, att, meta = ev.run_model(model, ds, device)
    pts = []
    for tau in ev.CURVE_TAU:
        sel = ev.select_from_probs(P, meta["primary"].astype(int), tau, env.cfg.aux_max)
        r = ev.metrics(env, sel, att, meta)
        pts.append(dict(param=tau, **{k: r[k] for k in ("mean_aux", "recall", "precision", "flicker")}))
    return pts


def load_model(ckpt, device):
    st = torch.load(ckpt, map_location=device, weights_only=False)
    m = ViewSelector(ModelConfig(**st["mcfg"])).to(device)
    m.load_state_dict(st["model"])
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--config", default="configs/train_proposed.yaml")
    ap.add_argument("--ckpt", default=None, help="제안 모델 (기본: 최신 trainval 실행)")
    ap.add_argument("--extra_ckpt", default=None, help="함께 비교할 모델(예: A2)")
    ap.add_argument("--extra_name", default="A2")
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--out_dir", default="results/robustness")
    ap.add_argument("--plot_only", action="store_true", help="저장된 json으로 표·그림만 다시 만든다")
    ap.add_argument("--tag", default="", help="결과 파일 이름 꼬리표 (예: t10) — 다른 실험 결과를 덮어쓰지 않게")
    args = ap.parse_args()

    stem = os.path.join(args.out_dir, f"robustness_{args.split}" + (f"_{args.tag}" if args.tag else ""))
    if args.plot_only:
        with open(stem + ".json", encoding="utf-8") as f:
            out = json.load(f)
        write_report(out, args, stem)
        return 0

    base, lab, tc = load_cfgs(args.config)
    version = base["version"]
    env = ev.Env(base, lab, tc, version)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or ev.latest_ckpt(tc.get("exp_name", "proposed"), version)
    models = {"model": load_model(ckpt, device)}
    if args.extra_ckpt:
        models[args.extra_name] = load_model(args.extra_ckpt, device)
    # 규칙은 한 번만 만들어 과거 기동 추정 캐시를 조건 사이에 재사용한다
    rules = {"B2": RuleBased(copy.deepcopy(env.cfg), "oracle"),
             "B2c": RuleBased(copy.deepcopy(env.cfg), "causal", wrapper=env.w)}

    lab_df = env.df.set_index("sample_token")
    ds0 = env.dataset(args.split, "gt", "clean")
    toks = [ds0.scene_tokens[sc][pos] for sc, pos in ds0.flat]
    load = float(lab_df.loc[toks, "n_aux"].mean())
    print(f"[강건성] {version} · {args.split} · 기준 표시량 {load:.2f}장 · 모델 {ckpt}")

    conds = [("all", "clean")] + [(t, lv) for t in args.types.split(",") for lv in LEVELS[1:]]
    out = dict(split=args.split, load=load, ckpt=ckpt, extra=args.extra_ckpt, results=[])
    for typ, lv in conds:
        if lv == "clean":
            ds = env.dataset(args.split, "gt", "clean")
        else:
            ds = env.dataset(args.split, "noisy", f"{lv}:{typ}")
        row = dict(type=typ, level=lv, curves={})
        for name, rule in rules.items():
            row["curves"][name] = rule_curve(env, rule, ds)
        for name, m in models.items():
            row["curves"][name] = model_curve(env, m, ds, device)
        row["at_load"] = {k: dict(recall=interp(v, "recall", load),
                                  precision=interp(v, "precision", load),
                                  flicker=interp(v, "flicker", load))
                          for k, v in row["curves"].items()}
        out["results"].append(row)
        s = "  ".join(f"{k} {v['recall']:.1f}%" if v["recall"] is not None else f"{k} —"
                      for k, v in row["at_load"].items())
        print(f"  {typ:5s} {lv:6s}  {s}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    write_report(out, args, stem)
    return 0


def write_report(out, args, stem):
    load, ckpt = out["load"], out["ckpt"]
    # ---- 표 ----
    names = list(out["results"][0]["at_load"].keys())
    clean = out["results"][0]["at_load"]
    lines = [f"# 강건성 — {out['split']} 분할, 같은 표시량({load:.2f}장)에서의 위험 사건 포착률\n",
             f"모델: `{ckpt}`" + (f" · {args.extra_name}: `{out['extra']}`" if out.get("extra") else "") + "\n"]
    for typ in args.types.split(","):
        rows = [out["results"][0]] + [r for r in out["results"] if r["type"] == typ]
        lines.append(f"\n## {TYPE_KO[typ]} ({typ})\n")
        lines.append("| 강도 | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for r in rows:
            cells = []
            for n in names:
                v = r["at_load"][n]["recall"]
                cells.append("—" if v is None else f"{v:.1f}%")
            lines.append(f"| {r['level']} | " + " | ".join(cells) + " |")
        heavy = next(r for r in rows if r["level"] == "heavy")
        drops = []
        for n in names:
            a, b = clean[n]["recall"], heavy["at_load"][n]["recall"]
            drops.append("—" if a is None or b is None else f"{b - a:+.1f}%p")
        lines.append("| **열화 (clean→heavy)** | " + " | ".join(drops) + " |")
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 그림 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use_korean_font()
    types = args.types.split(",")
    fig, axes = plt.subplots(1, len(types), figsize=(4.2 * len(types), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    style = {"B2": ("#9aa4ab", "--"), "B2c": ("#5a646c", "-"), "model": ("#0d6b73", "-"),
             args.extra_name: ("#b8422a", ":")}
    for ax, typ in zip(axes, types):
        rows = [out["results"][0]] + [r for r in out["results"] if r["type"] == typ]
        for n in names:
            ys = [r["at_load"][n]["recall"] for r in rows]
            c, ls = style.get(n, ("#333", "-"))
            ax.plot(range(len(rows)), [np.nan if y is None else y for y in ys],
                    marker="o", color=c, linestyle=ls, label=n, linewidth=2)
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(["clean", "light", "medium", "heavy"] if typ != "drop" else ["0대", "1대", "2대", "3대"])
        ax.set_title(TYPE_KO[typ])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"포착률 @ {load:.2f}장 (%)")
    axes[-1].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=150)
    print(f"\n✅ {stem}.json · .md · .png")


if __name__ == "__main__":
    raise SystemExit(main())
