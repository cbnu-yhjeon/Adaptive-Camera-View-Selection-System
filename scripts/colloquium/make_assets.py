"""INDAI 콜로키움 포스터·논문용 그림과 수치 (results/ 산출물에서만 읽는다 — 원칙 4).

산출 (results/colloquium/)
  numbers.json            포스터·논문 빌더가 쓰는 수치 모음 (출처 파일 포함)
  fig/*_poster.png        포스터 실제 크기(mm)·글자 크기로 그린 차트
  fig/*_paper.png         논문 단(85 mm) 크기 차트
  fig/concept_grid.jpg    6채널 격자 + 진행 방향·보조 뷰 표시 (골드셋 판정 장면)

골드셋 결과는 최종 평가(results/eval — scripts/run_final.sh가 만든다)를 쓴다.
강건성 그림은 손상 조건별 골드 Top-1 '수준'을 그린다 — 주장 3을 열화 기울기가 아니라
수준 차이로 서술하기로 한 결정(2026-10-13, 명세 E-4)과 맞춘다.

사용:
    python scripts/colloquium/make_assets.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "colloquium")
FIG = os.path.join(OUT, "fig")
GOLD_DIR = os.path.join(ROOT, "results", "eval")
GOLD_JSON = os.path.join(ROOT, "data", "goldset_v1.json")
UI_DIR = os.path.join(ROOT, "tools", "goldset_ui")
CONCEPT_TOKEN = "e961b547d451"     # 골드셋 판정 장면 (scene-0068, 좌측 지시등, 정답: 좌후·좌전)

FONT_DIRS = ["/mnt/c/Windows/Fonts", "/usr/share/fonts"]
INK, MUTED, RULE, GRID = "#14181c", "#5a646c", "#d8dee2", "#e6eaed"
TEAL, AMBER, GRAY, DARKGRAY, RUST = "#0d6b73", "#a86b00", "#9aa4ab", "#5a646c", "#b8422a"
MM = 1 / 25.4

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
KO = {"CAM_FRONT": "전방", "CAM_FRONT_RIGHT": "우전측방", "CAM_BACK_RIGHT": "우후측방",
      "CAM_BACK": "후방", "CAM_BACK_LEFT": "좌후측방", "CAM_FRONT_LEFT": "좌전측방"}


def setup_font():
    for d in FONT_DIRS:
        for f in ("malgun.ttf", "malgunbd.ttf"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                         "axes.edgecolor": RULE, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def load(path):
    with open(os.path.join(ROOT, path) if not os.path.isabs(path) else path, encoding="utf-8") as f:
        return json.load(f)


def interp_at(curve, load_, key="recall"):
    xs = np.array([p["mean_aux"] for p in curve])
    ys = np.array([p[key] for p in curve])
    o = np.argsort(xs)
    return float(np.interp(load_, xs[o], ys[o]))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ---------------------------------------------------------------------------
def collect() -> dict:
    N: dict = {"sources": {}}
    # 라벨
    rep = load("results/label_report.json")
    oc = load("results/operating_curve.json")
    adopt = next(p for p in oc["curve"] if abs(p["theta"] - oc["adopted"]) < 1e-9)
    N["label"] = dict(
        n_keyframes=rep["n_keyframes"], hazard_channels=rep["hazard"]["channels"],
        hazard_frames=rep["hazard"]["frames"], hazard_pct=rep["hazard"]["frame_pct"],
        test_hazard_frames=rep["hazard"]["by_split"]["test"],
        mean_aux=adopt["load"], recall=adopt["recall"], precision=adopt["precision"],
        zero=adopt["zero"], one=adopt["one"], two=adopt["two"],
        always1=oc["always_one"],
        signal={k: v["same_pct"] for k, v in rep["signal"]["by_maneuver"].items() if v["same_pct"] is not None},
        n_split=rep["hazard"]["n_by_split"], maneuver=rep["labels"]["maneuver"],
        curve=[dict(theta=p["theta"], load=p["load"], recall=p["recall"], precision=p["precision"]) for p in oc["curve"]])
    N["sources"]["label"] = ["results/label_report.json", "results/operating_curve.json"]

    # 표 1 (골드셋 중간 + 같은 표시량 포착률)
    curves = load(os.path.join(GOLD_DIR, "curve_test_noisy-medium.json"))
    load_ = curves["load"]
    gold = load(GOLD_JSON)
    methods = [("b1", "B1 고정"), ("b2", "B2 규칙(미래 기동)"), ("b2c", "B2c 규칙(과거만)"),
               ("b3", "B3 영상만"), ("b4", "B4 단일 프레임"), ("model", "제안 모델"), ("label", "의사 라벨(참고)")]
    T = []
    hits = {}
    for key, name in methods:
        r = load(os.path.join(GOLD_DIR, f"{key}_test_noisy-medium.json"))
        g = r.get("gold", {})
        u, w = g.get("unweighted", {}), g.get("weighted", {})
        n = u.get("n_nonempty", 0)
        has = key != "b1" and u.get("top1") is not None
        k = round(u["top1"] * n) if has and n else 0
        lo, hi = wilson(k, n) if has else (None, None)
        hits[key] = g.get("top1_hit", {})
        T.append(dict(key=key, name=name, n_gold=g.get("n"), n_nonempty=n,
                      top1=u.get("top1") if key != "b1" else None,
                      top1_ci=(lo, hi), top2=u.get("top2") if key != "b1" else None,
                      top1_w=w.get("top1") if key != "b1" else None,
                      quiet_u=u.get("quiet"), n_none=(g.get("n", 0) - n) if g else None,
                      mean_aux=r["mean_aux"], recall=r["recall"], precision=r["precision"],
                      flicker=r["flicker"],
                      recall_at_load=(interp_at(curves["curves"][key], load_) if key in curves["curves"] else None),
                      flicker_at_load=(interp_at(curves["curves"][key], load_, "flicker") if key in curves["curves"] else None)))
    for row in T:
        if row["key"] in ("b1", "model") or not hits.get(row["key"]):
            continue
        common = [t for t in hits["model"] if t in hits[row["key"]]]
        b = sum(1 for t in common if hits["model"][t] and not hits[row["key"]][t])
        c = sum(1 for t in common if hits[row["key"]][t] and not hits["model"][t])
        row["vs_model"] = dict(model_only=b, other_only=c, p=mcnemar_exact(b, c))
    N["table1"] = dict(load=load_, n_labeled=gold["meta"]["n_done"], n_total=gold["meta"]["n_items"],
                       saved_at=gold["meta"]["saved_at"], rows=T)
    N["sources"]["table1"] = [os.path.relpath(GOLD_DIR, ROOT) + "/*_test_noisy-medium.json",
                              "data/goldset_v1.json"]

    # ablation
    ab = load("results/ablation.json")
    N["ablation"] = dict(load=ab["load"], rows=[
        dict(tag=r["tag"], desc=r["desc"], val_ap=r["val_ap"],
             test_ap=r["res"]["noisy-medium"]["ap"],
             recall_clean=r["res"]["gt"]["at_load"]["recall"],
             recall_noisy=r["res"]["noisy-medium"]["at_load"]["recall"],
             precision_noisy=r["res"]["noisy-medium"]["at_load"]["precision"],
             flicker_noisy=r["res"]["noisy-medium"]["at_load"]["flicker"]) for r in ab["rows"]],
        rules=[dict(tag=r["tag"], recall_clean=r["res"]["gt"]["at_load"]["recall"],
                    recall_noisy=r["res"]["noisy-medium"]["at_load"]["recall"]) for r in ab["rules"]])
    N["sources"]["ablation"] = ["results/ablation.json"]

    # 강건성 (test)
    rb = load("results/robustness/robustness_test.json")
    clean = next(x for x in rb["results"] if x["level"] == "clean")
    R = {"load": rb["load"], "ckpt": rb["ckpt"], "clean": {k: v["recall"] for k, v in clean["at_load"].items()}}
    for typ in ("miss", "fp", "pos", "drop"):
        R[typ] = {lv: {k: v["recall"] for k, v in x["at_load"].items()}
                  for x in rb["results"] if x["type"] == typ for lv in [x["level"]]}
    # 골드 Top-1 (비가중 — 정답이 있는 장면 기준) — 손상 조건별 수준
    conds = [("all", "clean")] + [(t, "heavy") for t in ("miss", "fp", "pos", "drop")]
    G = {}
    for typ, lv in conds:
        x = next(r for r in rb["results"] if r["type"] == typ and r["level"] == lv)
        G[typ] = {k: v["top1_unweighted"] for k, v in x["gold"].items()}
    R["gold"] = G
    R["gold_n"] = next(iter(clean["gold"].values()))["n"]
    gd = [100 * (G[t]["model"] - G[t]["B2c"]) for t, _ in conds]
    rd = [(R[t]["heavy"]["model"] - R[t]["heavy"]["B2c"]) if t != "all" else (R["clean"]["model"] - R["clean"]["B2c"])
          for t, _ in conds]
    deg = lambda k: [100 * (G[t][k] - G["all"][k]) for t, _ in conds[1:]]
    R["claim3"] = dict(gold_diff=(min(gd), max(gd)), recall_diff=(min(rd), max(rd)),
                       deg_model=(min(deg("model")), max(deg("model"))), deg_b2c=(min(deg("B2c")), max(deg("B2c"))),
                       a2_gold_clean=G["all"].get("A2"))
    N["robust"] = R
    N["sources"]["robust"] = ["results/robustness/robustness_test.json"]
    N["setup"] = setup(rb["ckpt"])
    N["sources"]["setup"] = ["configs/train_proposed.yaml", "configs/label_v1.yaml", rb["ckpt"]]
    return N


def setup(ckpt: str) -> dict:
    """실험 환경 — 설정 파일·체크포인트·실행 장치에서 읽는다(논문 표 1)."""
    import torch
    sys.path.insert(0, ROOT)
    from src.models.proposed import ModelConfig, ViewSelector
    from src.utils.config import load_cfgs
    base, lab, tc = load_cfgs(os.path.join(ROOT, "configs", "train_proposed.yaml"))
    st = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    model = ViewSelector(ModelConfig(**st["mcfg"]))
    met = load(os.path.join(os.path.dirname(ckpt), "metrics.json"))
    tr, lo = tc["train"], tc.get("loss", {})
    return dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                torch=torch.__version__.split("+")[0],
                params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                best_epoch=met.get("best_epoch"), val_ap=met.get("best_val_ap", met.get("best_f1")),
                seq_len=tr.get("seq_len"), lr=tr.get("lr"), batch=tr.get("batch_size"), epochs=tr.get("epochs"),
                patience=tr.get("patience"), noise_mix=tr.get("noise_mix"), seed=base.get("seed", 42),
                tf_layers=tc["model"].get("tf_layers"), d=tc["model"].get("d"))


# ---------------------------------------------------------------------------
def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_size(kind):
    # (가로 mm, 세로 mm, 기본 글자 pt)
    return {"poster": (200, 118, 21), "paper": (84, 52, 7.2)}[kind]


def chart_opcurve(N, kind):
    w, h, fs = fig_size(kind)
    plt.rcParams["font.size"] = fs
    fig, ax = plt.subplots(figsize=(w * MM, h * MM))
    cv = sorted(N["label"]["curve"], key=lambda p: p["load"])
    x = [p["load"] for p in cv]
    lw = 3.2 if kind == "poster" else 1.3
    ms = 9 if kind == "poster" else 3.4
    ax.plot(x, [p["recall"] for p in cv], "-o", color=TEAL, lw=lw, ms=ms, label="위험 포착률")
    ax.plot(x, [p["precision"] for p in cv], "-o", color=RUST, lw=lw, ms=ms, label="위험 슬롯 정밀도")
    L = N["label"]
    a1 = L["always1"]
    ax.plot([L["mean_aux"]] * 2, [L["recall"], L["precision"]], "o", ms=ms * 1.9, mfc="none", mec=INK, mew=lw * 0.7)
    ax.plot([a1["load"]] * 2, [a1["recall"], a1["precision"]], "s", ms=ms * 1.3, mfc="white", mec=MUTED, mew=lw * 0.7)
    ax.annotate(f"채택 θ=0.12\n{L['mean_aux']:.2f}장 · {L['recall']:.1f}% · {L['precision']:.1f}%",
                (L["mean_aux"], L["recall"]), xytext=(L["mean_aux"] + 0.02, 62), fontsize=fs * 0.92,
                color=INK, ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=lw * 0.4))
    ax.annotate(f"항상 1장\n{a1['recall']:.1f}% · {a1['precision']:.1f}%", (a1["load"], a1["precision"]),
                xytext=(a1["load"] - 0.01, 30), fontsize=fs * 0.92, color=MUTED, ha="right",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=lw * 0.4))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 104)
    ax.set_xlabel("평균 보조 뷰 수 (장)")
    ax.set_ylabel("%")
    ax.legend(loc="center left", frameon=False, fontsize=fs * 0.9, bbox_to_anchor=(0.0, 0.40))
    style_axes(ax)
    fig.tight_layout(pad=0.4)
    return fig


def chart_gold(N, kind):
    w, h, fs = fig_size(kind)
    plt.rcParams["font.size"] = fs
    fig, ax = plt.subplots(figsize=(w * MM, h * MM))
    rows = [r for r in N["table1"]["rows"] if r["top1"] is not None]
    order = ["label", "b2", "b4", "model", "b3", "b2c"]
    rows = sorted(rows, key=lambda r: order.index(r["key"]))[::-1]
    names = {"label": "의사 라벨(참고)", "b2": "B2 규칙·미래 기동", "b4": "B4 단일 프레임",
             "model": "제안 모델", "b3": "B3 영상만", "b2c": "B2c 규칙·과거만"}
    col = {"model": TEAL, "b2c": AMBER}
    y = np.arange(len(rows))
    vals = [100 * r["top1"] for r in rows]
    lo = [100 * (r["top1"] - r["top1_ci"][0]) for r in rows]
    hi = [100 * (r["top1_ci"][1] - r["top1"]) for r in rows]
    bh = 0.6
    ax.barh(y, vals, height=bh, color=[col.get(r["key"], GRAY) for r in rows],
            xerr=[lo, hi], error_kw=dict(ecolor=DARKGRAY, lw=1.6 if kind == "poster" else 0.7,
                                         capsize=6 if kind == "poster" else 2))
    for yi, v, r in zip(y, vals, rows):
        ax.text(2, yi, f"{v:.1f}%", va="center", ha="left", color="white", fontsize=fs * 0.9, fontweight="bold")
    ax.set_yticks(y, [names[r["key"]] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("골드셋 Top-1 (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    return fig


def chart_robust(N, kind):
    """손상 조건별 골드 Top-1 수준 — clean과 4가지 heavy 손상에서 제안 vs 규칙."""
    w, h, fs = fig_size(kind)
    plt.rcParams["font.size"] = fs
    fig, ax = plt.subplots(figsize=(w * MM, h * MM))
    G = N["robust"]["gold"]
    conds = [("all", "clean"), ("miss", "누락"), ("fp", "오검출"), ("pos", "위치\n오차"), ("drop", "카메라\n고장")]
    meths = [("B2", "B2 (미래 기동)", GRAY), ("B2c", "B2c (과거만)", AMBER), ("model", "제안 모델", TEAL)]
    x = np.arange(len(conds))
    bw = 0.26
    for i, (k, lab, c) in enumerate(meths):
        v = [100 * G[t][k] for t, _ in conds]
        bars = ax.bar(x + (i - 1) * bw, v, width=bw * 0.92, color=c, label=lab)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val - 1.5, f"{val:.0f}", ha="center", va="top",
                    fontsize=fs * 0.68, color="white", fontweight="bold", rotation=90 if kind == "paper" else 0)
    ax.set_xticks(x, [t for _, t in conds])
    ax.set_ylabel(f"골드셋 Top-1 (%, {N['robust']['gold_n']}장)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80])          # 100 눈금은 범례와 겹친다
    ax.legend(frameon=False, fontsize=fs * 0.85, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))
    style_axes(ax)
    fig.tight_layout(pad=0.4)
    return fig


def chart_ablation(N, kind):
    w, h, fs = fig_size(kind)
    plt.rcParams["font.size"] = fs
    fig, ax = plt.subplots(figsize=(w * MM, h * MM))
    rows = N["ablation"]["rows"]
    base = next(r for r in rows if r["tag"] == "제안")
    lab = {"A1": "A1 – CAN·지시등", "A2": "A2 – 위험도 입력", "A3": "A3 – GRU (1프레임)",
           "A4": "A4 – 화면 간 어텐션", "A5": "A5 – 시간 일관성 손실", "B3": "B3 영상만", "T10": "T10 이력 5초"}
    keep = [r for r in rows if r["tag"] in lab and r["tag"] != "T10"]
    keep = sorted(keep, key=lambda r: r["recall_noisy"] - base["recall_noisy"])[::-1]
    d = [r["recall_noisy"] - base["recall_noisy"] for r in keep]
    y = np.arange(len(keep))
    ax.barh(y, d, height=0.6, color=[RUST if v < -2 else GRAY for v in d])
    for yi, v, r in zip(y, d, keep):
        txt = f"{v:+.1f}%p" if abs(v) >= 0.05 else "±0.0%p"
        if r["tag"] == "A3":
            txt += f"  · flicker {r['flicker_noisy']:.2f} (제안 {base['flicker_noisy']:.2f})"
        big = v < -8
        ax.text(-0.4 if big else min(v, 0) - 0.3, yi, txt, va="center", ha="right",
                fontsize=fs * 0.82, color="white" if big else INK, fontweight="bold" if big else None)
    ax.set_yticks(y, [lab[r["tag"]] for r in keep])
    ax.set_xlim(-22.5, 0.8)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("제안 대비 위험 포착률 변화 (%p)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    return fig


# ---------------------------------------------------------------------------
def concept_grid(token_prefix: str, gold_sel: list[int], out: str):
    """골드셋 판정 장면의 6채널 격자 — 진행 방향(청록)·보조 뷰(적갈) 표시."""
    s = open(os.path.join(UI_DIR, "manifest.js"), encoding="utf-8").read()
    j = json.loads(s[s.index("{"):s.rindex("}") + 1])
    it = next(x for x in j["items"] if x["token"].startswith(token_prefix))
    fr = it["frames"][it["now_index"]]
    grid = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    tiles = [Image.open(os.path.join(UI_DIR, fr["img"][c])).convert("RGB") for c in grid]
    tw, th = tiles[0].size
    gap, bd = 10, 14
    W, Hh = 3 * tw + 2 * gap, 2 * th + gap
    canvas = Image.new("RGB", (W, Hh), "white")
    fdir = next(d for d in FONT_DIRS if os.path.exists(os.path.join(d, "malgunbd.ttf")))
    font = ImageFont.truetype(os.path.join(fdir, "malgunbd.ttf"), 34)
    small = ImageFont.truetype(os.path.join(fdir, "malgun.ttf"), 26)
    primary = CAMS[it["primary"]]
    aux = {CAMS[i] for i in gold_sel}
    for i, (c, im) in enumerate(zip(grid, tiles)):
        x, y = (i % 3) * (tw + gap), (i // 3) * (th + gap)
        canvas.paste(im, (x, y))
        d = ImageDraw.Draw(canvas)
        color = TEAL if c == primary else (RUST if c in aux else None)
        d.rectangle([x, y + th - 54, x + tw, y + th], fill=(20, 24, 28))
        d.text((x + bd + 10, y + th - 52), f"{KO[c]} · {c.replace('CAM_', '')}", font=small, fill="white")
        if color:
            for k in range(bd):
                d.rectangle([x + k, y + k, x + tw - 1 - k, y + th - 1 - k], outline=color)
            tag = "진행 방향 · 상시 표시" if c == primary else "보조 뷰 강조"
            tb = d.textbbox((0, 0), tag, font=font)
            d.rectangle([x + bd, y + bd, x + bd + tb[2] + 28, y + bd + tb[3] + 18], fill=color)
            d.text((x + bd + 14, y + bd + 6), tag, font=font, fill="white")
    canvas.save(out, quality=90)
    return dict(scene=it["scene"], speed_kmh=it["speed_kmh"], signal=it["signal"], token=it["token"],
                primary=it["primary"], aux=sorted(gold_sel))


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    setup_font()
    N = collect()
    gold = load(GOLD_JSON)["gold"]
    tok = next(t for t in gold if t.startswith(CONCEPT_TOKEN))
    N["concept"] = concept_grid(CONCEPT_TOKEN, gold[tok], os.path.join(FIG, "concept_grid.jpg"))
    for name, fn in [("opcurve", chart_opcurve), ("gold", chart_gold),
                     ("robust", chart_robust), ("ablation", chart_ablation)]:
        for kind in ("poster", "paper"):
            fig = fn(N, kind)
            fig.savefig(os.path.join(FIG, f"{name}_{kind}.png"), dpi=300 if kind == "poster" else 600)
            plt.close(fig)
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(N, f, ensure_ascii=False, indent=1)
    print("✅", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
