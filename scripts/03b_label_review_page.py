"""라벨 검수 페이지 생성 (M2 게이트) — 그림 + 판정 근거를 한 화면에.

03_visualize_labels.py가 만든 PNG에 기동 판정의 수치 근거(누적 헤딩 변화·선회 반경·
횡변위)를 붙여 하나의 HTML로 묶는다. 이미지를 base64로 심으므로 파일 하나만 열면 되고,
경로 재사용으로 옛 그림이 남는 사고가 나지 않는다.

사용:
    python scripts/03b_label_review_page.py
    python scripts/03b_label_review_page.py --viz results/label_viz --out results/label_review/index.html
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.labeling.label_function import LabelConfig  # noqa: E402
from src.labeling.maneuver import wrap_angle  # noqa: E402
from src.utils.constants import CAMERAS  # noqa: E402


def evidence(w, cfg, token):
    """기동 판정의 수치 근거 — 이 값들이 규칙의 입력이다."""
    sc = w.sample_scene_token(token)
    toks = w.sample_tokens_in_scene(sc)
    t0 = w.sample_timestamp(token)
    back = cfg.maneuver_window_back_s if cfg.maneuver_window_back_s is not None else cfg.maneuver_window_s
    fwd = cfg.maneuver_window_fwd_s if cfg.maneuver_window_fwd_s is not None else cfg.maneuver_window_s
    win = [tk for tk in toks if -back <= w.sample_timestamp(tk) - t0 <= fwd]
    if len(win) < 3:
        return None
    yaws = np.array([w.get_ego_yaw(tk) for tk in win])
    ts = np.array([w.sample_timestamp(tk) for tk in win])
    pos = np.array([w.get_ego_pose(tk)["translation"][:2] for tk in win])
    dev = np.degrees([wrap_angle(y - yaws[0]) for y in yaws])
    span = float(ts[-1] - ts[0])
    dist = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    v = dist / max(span, 1e-6)
    R = v / (abs(np.radians(dev[-1])) / max(span, 1e-6)) if abs(dev[-1]) > 0.5 else float("inf")
    h = np.array([np.cos(yaws[0]), np.sin(yaws[0])])
    n = np.array([-h[1], h[0]])
    lat = float(((pos - pos[0]) @ n)[-1])
    return dict(scene=w.scene_name(sc), net=float(dev[-1]), R=R, lat=lat,
                speed=v, span=span, n_win=len(win))


def img_uri(path, width=1080, quality=78):
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
:root{--paper:#f4f6f7;--surface:#fff;--surface-2:#eef1f3;--ink:#14181c;--muted:#5a646c;
--rule:#d8dee2;--accent:#0d6b73;--accent-bg:#e2f0f0;--signal:#b8422a;--signal-bg:#f7e7e1;
--f:"Malgun Gothic","맑은 고딕",system-ui,sans-serif;--m:Consolas,monospace;}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:var(--f);font-size:15px;line-height:1.7}
.wrap{max-width:1180px;margin:0 auto;padding:40px 22px 90px}
h1{font-size:26px;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 22px}
.note{border:1px solid var(--accent);background:var(--accent-bg);padding:14px 16px;margin:18px 0 26px}
.note b{color:var(--accent)}
.card{background:var(--surface);border:1px solid var(--rule);margin:0 0 22px}
.card.flag{border-color:var(--signal)}
.hd{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline;
padding:12px 16px;border-bottom:1px solid var(--rule);background:var(--surface-2)}
.no{font-family:var(--m);font-size:12px;color:var(--muted)}
.mv{font-weight:700;font-size:16px}
.mv.turn{color:var(--accent)}.mv.lc{color:var(--signal)}
.sc{font-family:var(--m);font-size:12px;color:var(--muted)}
.ev{margin-left:auto;font-family:var(--m);font-size:12px;color:var(--muted)}
.ev b{color:var(--ink)}
.card img{display:block;width:100%;height:auto}
.ft{padding:10px 16px;border-top:1px solid var(--rule);font-size:13px;color:var(--muted)}
.ft b{color:var(--ink)}
table{border-collapse:collapse;width:100%;margin:10px 0 0;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule)}
th{font-family:var(--m);font-size:11.5px;color:var(--muted);font-weight:400}
@media(prefers-color-scheme:dark){:root{--paper:#0f1316;--surface:#161b1f;--surface-2:#1c2227;
--ink:#e6eaec;--muted:#98a3ab;--rule:#29323a;--accent:#4fbac1;--accent-bg:#11302f;
--signal:#e08063;--signal-bg:#35211a}}

/* 판정 입력 */
.bar{position:sticky;top:0;z-index:9;background:var(--surface);border:1px solid var(--rule);
padding:10px 14px;margin:0 0 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.bar .cnt{font-family:var(--m);font-size:13px}
.bar .cnt b{font-size:15px}
.bar button{font:inherit;font-size:13px;padding:5px 12px;border:1px solid var(--rule);
background:var(--surface-2);color:var(--ink);cursor:pointer}
.bar button:hover{border-color:var(--accent)}
.vd{display:flex;gap:8px;align-items:center;padding:10px 16px;border-top:1px solid var(--rule);
flex-wrap:wrap}
.vd button{font:inherit;font-size:13px;padding:5px 14px;border:1px solid var(--rule);
background:var(--surface);color:var(--muted);cursor:pointer}
.vd button:hover{border-color:var(--accent);color:var(--ink)}
.vd button.on[data-v="ok"]{background:var(--accent);border-color:var(--accent);color:#fff}
.vd button.on[data-v="bad"]{background:var(--signal);border-color:var(--signal);color:#fff}
.vd button.on[data-v="hmm"]{background:var(--muted);border-color:var(--muted);color:#fff}
.vd input{flex:1;min-width:220px;font:inherit;font-size:13px;padding:5px 9px;
border:1px solid var(--rule);background:var(--surface);color:var(--ink)}
.card.done{border-color:var(--accent)}
.card.bad{border-color:var(--signal)}
#out{width:100%;height:150px;font-family:var(--m);font-size:12px;margin-top:10px;
border:1px solid var(--rule);background:var(--surface);color:var(--ink);padding:10px;display:none}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", default="results/label_viz")
    ap.add_argument("--out", default="results/label_review/index.html")
    ap.add_argument("--version", default=None)
    args = ap.parse_args()

    d = os.path.join(os.path.dirname(__file__), "..", "configs")
    base = yaml.safe_load(open(os.path.join(d, "base.yaml"), encoding="utf-8"))
    lab = yaml.safe_load(open(os.path.join(d, "label_v1.yaml"), encoding="utf-8"))
    cfg = LabelConfig.from_dict(lab)
    version = args.version or base["version"]

    df = pd.read_parquet(f"data/labels_{version}.parquet")
    w = NuscWrapper(base["dataroot"], version,
                    require_image_files=base.get("require_image_files", True))

    files = sorted(f for f in os.listdir(args.viz) if f.endswith(".png"))
    th = cfg.maneuver_th
    cards, rows = [], []
    for fn in files:
        m = re.match(r"viz_(\d+)_(.+)_([0-9a-f]{8})\.png", fn)
        if not m:
            continue
        no, mv, pref = m.group(1), m.group(2), m.group(3)
        hit = df[df.sample_token.str.startswith(pref)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        ev = evidence(w, cfg, r["sample_token"])
        aux = [i for i in range(6) if int(r.get(f"aux_{i}", 0))]
        cls = "turn" if mv.startswith("TURN") else ("lc" if mv.startswith("LANE") else "")
        Rtxt = "∞" if not ev or not np.isfinite(ev["R"]) else f"{ev['R']:.0f} m"
        evtxt = ("" if not ev else
                 f"Δψ <b>{ev['net']:+.1f}°</b> · 반경 <b>{Rtxt}</b> · "
                 f"횡변위 <b>{ev['lat']:+.1f} m</b> · {ev['speed']:.1f} m/s")
        aux_txt = ", ".join(f"[{i}] {CAMERAS[i]}" for i in aux) or "없음"
        ttc = f"{r['min_ttc']:.1f}s" if r["min_ttc"] >= 0 else "없음"
        flag = " flag" if mv.startswith(("TURN", "LANE")) else ""
        cards.append(f"""<div class="card{flag}">
  <div class="hd"><span class="no">{no}</span>
    <span class="mv {cls}">{mv}</span>
    <span class="sc">{ev['scene'] if ev else ''} · {pref}</span>
    <span class="ev">{evtxt}</span></div>
  <img src="{img_uri(os.path.join(args.viz, fn))}" alt="{fn}">
  <div class="ft">진행방향 <b>{CAMERAS[int(r['primary_view'])]}</b> ·
    보조 <b>{len(aux)}장</b> — {aux_txt} · 객체 {int(r['n_objects'])}개 · 최소 TTC {ttc}</div>
  <div class="vd" data-id="{no}">
    <button data-v="ok">맞음</button>
    <button data-v="bad">틀림</button>
    <button data-v="hmm">애매</button>
    <input type="text" placeholder="무엇이 어떻게 틀렸는지 (선택)">
  </div>
</div>""")
        rows.append((no, mv, ev, r))

    trs = []
    for no, mv, ev, r in rows:
        if ev is None:
            net_txt = r_txt = lat_txt = "—"
        else:
            net_txt = f"{ev['net']:+.1f}°"
            r_txt = "∞" if not np.isfinite(ev["R"]) else f"{ev['R']:.0f}"
            lat_txt = f"{ev['lat']:+.1f}"
        trs.append(f"<tr><td>{no}</td><td>{mv}</td><td>{net_txt}</td>"
                   f"<td>{r_txt}</td><td>{lat_txt}</td><td>{int(r['n_aux'])}</td></tr>")

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>라벨 검수 — {version}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>의사 라벨 검수 (M2 게이트)</h1>
<p class="sub">{version} · {len(df):,} keyframe 중 층화 추출 {len(rows)}장 ·
기동 판정 규칙 수정 후 재생성</p>

<div class="note">
<p><b>무엇을 봐야 하는가</b> — 기동 라벨이 영상 속 실제 상황과 맞는지입니다.
특히 <b>TURN</b>과 <b>LANE_CHANGE</b>(붉은 테두리)는 보조 뷰를 무조건 켜므로,
틀리면 그대로 오탐이 됩니다.</p>
<p>각 카드 우측의 <b>Δψ</b>(누적 헤딩 변화) · <b>반경</b> · <b>횡변위</b>가 판정의 근거입니다.
교차로 회전은 반경 {th.turn_radius_max:.0f} m 이하, 차선변경은 횡변위
{th.lane_change_lat_min:.2f} m 이상이어야 합니다. 값이 기준에 맞는데 영상이 다르게 보이면
알려주세요 — 규칙이 아니라 신호 자체를 봐야 합니다.</p>
<p>보조 뷰(빨간 막대)에 실제로 봐야 할 것이 있는지도 함께 봐주세요.</p>
</div>

<table><thead><tr><th>번호</th><th>기동</th><th>누적 Δψ</th><th>선회반경(m)</th>
<th>횡변위(m)</th><th>보조 뷰</th></tr></thead><tbody>{''.join(trs)}</tbody></table>

<div class="bar">
  <span class="cnt">판정 <b id="n">0</b>/{len(rows)} ·
    맞음 <b id="nok">0</b> · 틀림 <b id="nbad">0</b> · 애매 <b id="nhmm">0</b></span>
  <button id="copy">결과 복사</button>
  <button id="show">결과 보기</button>
  <button id="clear">초기화</button>
</div>
<textarea id="out" readonly></textarea>

{''.join(cards)}
</div>
<script>
const KEY = "label_review_{version}";
const load = () => {{ try {{ return JSON.parse(localStorage.getItem(KEY)) || {{}}; }}
                      catch (e) {{ return {{}}; }} }};
const save = (v) => {{ try {{ localStorage.setItem(KEY, JSON.stringify(v)); }} catch (e) {{}} }};
let V = load();

function paint() {{
  let ok = 0, bad = 0, hmm = 0;
  document.querySelectorAll(".vd").forEach(row => {{
    const id = row.dataset.id, cur = V[id];
    row.querySelectorAll("button").forEach(b =>
      b.classList.toggle("on", !!cur && cur.v === b.dataset.v));
    row.querySelector("input").value = (cur && cur.note) || "";
    const card = row.closest(".card");
    card.classList.toggle("done", !!cur && cur.v === "ok");
    card.classList.toggle("bad", !!cur && cur.v === "bad");
    if (cur) {{ if (cur.v === "ok") ok++; else if (cur.v === "bad") bad++; else hmm++; }}
  }});
  document.getElementById("n").textContent = ok + bad + hmm;
  document.getElementById("nok").textContent = ok;
  document.getElementById("nbad").textContent = bad;
  document.getElementById("nhmm").textContent = hmm;
}}

document.querySelectorAll(".vd").forEach(row => {{
  const id = row.dataset.id;
  row.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {{
    const cur = V[id] || {{}};
    V[id] = (cur.v === b.dataset.v) ? undefined : {{ v: b.dataset.v, note: cur.note || "" }};
    if (!V[id]) delete V[id];
    save(V); paint();
  }}));
  row.querySelector("input").addEventListener("input", e => {{
    V[id] = Object.assign({{ v: "hmm" }}, V[id], {{ note: e.target.value }});
    save(V); paint();
  }});
}});

const asText = () => Object.keys(V).sort().map(k =>
  `${{k}}  ${{ {{ok: "맞음", bad: "틀림", hmm: "애매"}}[V[k].v] }}` +
  (V[k].note ? `  — ${{V[k].note}}` : "")).join("\\n") || "(판정 없음)";

document.getElementById("show").addEventListener("click", () => {{
  const o = document.getElementById("out");
  o.value = asText();
  o.style.display = o.style.display === "block" ? "none" : "block";
}});
document.getElementById("copy").addEventListener("click", async () => {{
  const t = asText();
  try {{ await navigator.clipboard.writeText(t); }}
  catch (e) {{ const o = document.getElementById("out");
               o.value = t; o.style.display = "block"; o.select(); }}
  const b = document.getElementById("copy");
  b.textContent = "복사됨"; setTimeout(() => b.textContent = "결과 복사", 1200);
}});
document.getElementById("clear").addEventListener("click", () => {{
  if (confirm("판정을 모두 지울까요?")) {{ V = {{}}; save(V); paint(); }}
}});
paint();
</script>
</body></html>"""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ {args.out}  ({os.path.getsize(args.out)/1024/1024:.1f} MB, {len(rows)}장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
