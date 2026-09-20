"""연구계획서의 시각자료를 개별 파일로 추출한다.

계획서 HTML(아티팩트 원본)에서 그림 블록을 떼어내 다음을 만든다.
  · PNG — Chrome 헤드리스로 2배 해상도 렌더링 후 여백 트림 (발표·문서 삽입용)
  · SVG — 차트·개념도는 벡터로도 저장 (논문 그림으로 확대해도 깨지지 않음)

CSS 변수는 독립 파일에서 동작하지 않으므로 라이트 모드 값으로 인라인한다.

사용:
    python scripts/export_figures.py --src <계획서.html> --out results/figures
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile

from PIL import Image

CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

# 라이트 모드 토큰 — 문서의 :root 값과 동일
TOKENS = """
  --paper:#f4f6f7; --surface:#ffffff; --surface-2:#eef1f3;
  --ink:#14181c; --muted:#5a646c; --rule:#d8dee2; --rule-soft:#e6eaed;
  --accent:#0d6b73; --accent-bg:#e2f0f0; --signal:#b8422a; --signal-bg:#f7e7e1;
  --s-recall:#00929c; --s-prec:#bc4b35;
  --f-body:"Malgun Gothic","IBM Plex Sans KR",system-ui,sans-serif;
  --f-mono:"Consolas","IBM Plex Mono",monospace;
"""

# 추출할 그림: (파일명, 시작 앵커, 종료 앵커, 렌더 폭)
FIGURES = [
    ("fig1_console_layout", '<figure class="indent fig-block">\n      <div class="console">',
     "</figure>", 900),
    ("fig2_hazard_rarity", '<figure class="indent fig-block">\n      <div class="chart">\n        <svg viewBox="0 0 760 96"',
     "</figure>", 860),
    ("fig3_view_policy", '<figure class="indent">\n      <div class="diagram">\n        <svg viewBox="0 0 480 330"',
     "</figure>", 620),
    ("fig4_label_example", '<figure class="indent fig-block">\n      <div class="camgrid tight">',
     "</figure>", 780),
    ("fig5_operating_curve", '<figure class="indent fig-block">\n      <div class="chart">\n        <div class="legend">',
     "</figure>", 860),
]


def extract_css(src: str) -> str:
    """문서의 <style> 전체를 가져와 :root 토큰만 인라인 값으로 교체."""
    css = re.search(r"<style>(.*?)</style>", src, re.S).group(1)
    css = re.sub(r"@media \(prefers-color-scheme: dark\).*?\n  \}\n  \}", "", css, flags=re.S)
    css = re.sub(r":root\[data-theme=\"dark\"\].*?\n  \}", "", css, flags=re.S)
    return css


def block(src: str, start: str, end: str) -> str:
    """시작 앵커부터 짝이 맞는 닫는 태그까지. camgrid 안에는 <figure>가 중첩되어 있으므로
    단순히 첫 </figure>를 찾으면 카메라 한 칸에서 잘린다."""
    i = src.index(start)
    tag = end.strip("</>")
    close = f"</{tag}>"
    depth, j = 0, i
    while True:
        o = src.find(f"<{tag}", j)
        c = src.find(close, j)
        if c == -1:
            raise ValueError(f"닫는 {close} 없음")
        if o != -1 and o < c:
            depth += 1
            j = o + len(tag) + 1
        else:
            depth -= 1                      # 감소 후에 판정해야 바깥 태그에서 멈춘다
            j = c + len(close)
            if depth == 0:
                return src[i:j]


def standalone(css: str, body: str, width: int) -> str:
    # 문서 CSS를 먼저 넣고, 배경·여백 재정의를 뒤에 둔다
    # (문서의 body 배경이 남으면 트림이 여백을 구분하지 못한다)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>
:root {{{TOKENS}}}
{css}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; padding:0; background:#ffffff !important; }}
body {{ padding:16px; width:{width}px; font-family:var(--f-body); }}
.indent {{ padding-left:0 !important; }}
figure {{ margin:0; }}
</style></head><body>{body}</body></html>"""


def shoot(html: str, out_png: str, width: int) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     dir="/mnt/c/temp", encoding="utf-8") as f:
        f.write(html)
        tmp = f.name
    win_html = "C:\\temp\\" + os.path.basename(tmp)
    win_png = "C:\\temp\\_shot.png"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2",
                    f"--window-size={width},2400",
                    f"--screenshot={win_png}", f"file:///{win_html.replace(chr(92), '/')}"],
                   capture_output=True, timeout=180)
    import numpy as np
    im = Image.open("/mnt/c/temp/_shot.png").convert("RGB")
    a = np.asarray(im)
    # 흰 여백 트림 — 채널별로 판정한다(튜플 비교는 사전식이라 색을 잘못 본다)
    ink = (a < 247).any(axis=2)
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    if len(rows) and len(cols):
        pad = 12
        im = im.crop((max(0, cols.min() - pad), max(0, rows.min() - pad),
                      min(a.shape[1], cols.max() + pad), min(a.shape[0], rows.max() + pad)))
    im.save(out_png)
    os.remove(tmp)


def export_svg(body: str, css: str, out_svg: str) -> bool:
    """그림 블록 안의 <svg>를 독립 파일로. 클래스 규칙을 인라인 <style>로 심는다."""
    m = re.search(r"<svg .*?</svg>", body, re.S)
    if not m:
        return False
    svg = m.group(0)
    used = set(re.findall(r'class="([^"]+)"', svg))
    names = {n for group in used for n in group.split()}
    rules = []
    for name in sorted(names):
        for rule in re.findall(rf"(?:^|\n)\s*\.{re.escape(name)}\s*\{{([^}}]*)\}}", css):
            rules.append(f".{name} {{{rule.strip()}}}")
    for sel, body_css in re.findall(r"(?:^|\n)\s*(\.chart svg text|\.diagram svg text)\s*\{([^}]*)\}", css):
        rules.append(f"svg text {{{body_css.strip()}}}")
    style = "<style>\n:root {%s}\n%s\n</style>" % (TOKENS, "\n".join(rules))
    svg = svg.replace(">", ">\n" + style, 1)
    if "xmlns" not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    open(out_svg, "w", encoding="utf-8").write(svg)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    src = open(args.src, encoding="utf-8").read()
    css = extract_css(src)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("/mnt/c/temp", exist_ok=True)

    for name, start, end, width in FIGURES:
        body = block(src, start, end)
        png = os.path.join(args.out, f"{name}.png")
        shoot(standalone(css, body, width), png, width)
        size = os.path.getsize(png) / 1024
        svg_note = ""
        if name != "fig1_console_layout" and name != "fig4_label_example":
            svg = os.path.join(args.out, f"{name}.svg")
            if export_svg(body, css, svg):
                svg_note = f" + SVG {os.path.getsize(svg)/1024:.0f} KB"
        print(f"  {name}.png  {Image.open(png).size}  {size:.0f} KB{svg_note}")

    print(f"\n✅ {args.out}/ 에 저장 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
