"""원격주행 소개 슬라이드(1장) 생성 — 발표용 .pptx.

nuScenes 실촬영 6채널을 그대로 써서 "오퍼레이터가 보는 화면"을 보여준다.
도형·텍스트는 이미지가 아니라 네이티브 개체로 넣어 발표자가 직접 편집할 수 있게 한다.

사용:
    python scripts/make_teleop_slide.py            # 기본: scene-0538
    python scripts/make_teleop_slide.py --token 1a06922c --out results/slides/x.pptx

이미지 출처 표기(nuScenes, CC BY-NC-SA 4.0)는 라이선스상 필수이므로 슬라이드에 포함한다.
"""

from __future__ import annotations

import argparse
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.constants import CAMERAS  # noqa: E402

# ---- 색 (연구계획서·검수 페이지와 동일 계열) ----
INK = RGBColor(0x14, 0x18, 0x1C)
MUTED = RGBColor(0x5A, 0x64, 0x6C)
RULE = RGBColor(0xD8, 0xDE, 0xE2)
SURF2 = RGBColor(0xEE, 0xF1, 0xF3)
ACCENT = RGBColor(0x0D, 0x6B, 0x73)
SIGNAL = RGBColor(0xB8, 0x42, 0x2A)
SIGNAL_BG = RGBColor(0xF7, 0xE7, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

BODY = "맑은 고딕"
MONO = "Consolas"

GRID = [("CAM_FRONT_LEFT", "좌전측방"), ("CAM_FRONT", "전방"), ("CAM_FRONT_RIGHT", "우전측방"),
        ("CAM_BACK_LEFT", "좌후측방"), ("CAM_BACK", "후방"), ("CAM_BACK_RIGHT", "우후측방")]


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, *, size=12, bold=False, color=INK, font=BODY,
         space_after=0, line=None, first=False, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


def rule_line(slide, x, y, w, color=RULE, pt=0.75):
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Pt(pt))
    ln.fill.solid()
    ln.fill.fore_color.rgb = color
    ln.line.fill.background()
    ln.shadow.inherit = False
    return ln


def box(slide, shape, x, y, w, h, *, fill=SURF2, line=RULE, line_pt=1.0):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_pt)
    sh.shadow.inherit = False
    sh.text_frame.word_wrap = True
    return sh


def concept_diagram(slide, x0, y0):
    """차량 ⇄ 관제실 개념도 — 전부 편집 가능한 도형."""
    # 차량
    car = box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x0, y0 + 0.30, 1.05, 0.52)
    para(car.text_frame, "차량", size=11, bold=True, first=True, align=PP_ALIGN.CENTER)
    car.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    # 관제실
    ops = box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x0 + 3.05, y0 + 0.30, 1.05, 0.52)
    para(ops.text_frame, "관제실", size=11, bold=True, first=True, align=PP_ALIGN.CENTER)
    ops.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    # 업링크 (차량 → 관제실)
    up = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x0 + 1.15),
                                Inches(y0 + 0.36), Inches(1.80), Inches(0.17))
    up.fill.solid(); up.fill.fore_color.rgb = ACCENT
    up.line.fill.background(); up.shadow.inherit = False
    tf = textbox(slide, x0 + 1.05, y0 + 0.06, 2.0, 0.26)
    para(tf, "영상 업링크 · 6채널", size=9, color=ACCENT, font=MONO,
         first=True, align=PP_ALIGN.CENTER)

    # 다운링크 (관제실 → 차량)
    dn = slide.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, Inches(x0 + 1.15),
                                Inches(y0 + 0.62), Inches(1.80), Inches(0.17))
    dn.fill.solid(); dn.fill.fore_color.rgb = MUTED
    dn.line.fill.background(); dn.shadow.inherit = False
    tf = textbox(slide, x0 + 1.05, y0 + 0.82, 2.0, 0.26)
    para(tf, "제어 다운링크", size=9, color=MUTED, font=MONO,
         first=True, align=PP_ALIGN.CENTER)

    # 지연 배지
    bd = box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x0 + 1.33, y0 + 1.10, 1.45, 0.30,
             fill=SIGNAL_BG, line=SIGNAL, line_pt=0.75)
    bd.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(bd.text_frame, "왕복 지연 존재", size=9, color=SIGNAL, font=MONO,
         first=True, align=PP_ALIGN.CENTER)


def build(img_dir: str, out: str, scene: str, n_obj: int, min_ttc: float):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # 빈 레이아웃

    # 세로 구분선
    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.95), Inches(0.0),
                                 Pt(0.75), Inches(7.5))
    div.fill.solid(); div.fill.fore_color.rgb = RULE
    div.line.fill.background(); div.shadow.inherit = False

    # ---------------- 좌측: 정의 ----------------
    tf = textbox(slide, 0.55, 0.50, 4.1, 0.3)
    para(tf, "T E L E O P E R A T E D   D R I V I N G", size=9,
         color=ACCENT, font=MONO, first=True)

    tf = textbox(slide, 0.55, 0.78, 4.1, 0.9)
    para(tf, "원격주행", size=40, bold=True, first=True)

    tf = textbox(slide, 0.55, 1.62, 4.1, 0.7)
    para(tf, "자율주행이 멈춰 서는 상황을\n사람이 원격에서 인수한다",
         size=13, color=MUTED, line=1.35, first=True)

    tf = textbox(slide, 0.55, 2.45, 4.1, 1.5)
    para(tf, "차량은 도로 위에, 운전자는 관제실에 있다. 차량의 카메라 영상이 "
             "통신망으로 올라오고, 오퍼레이터의 조향·가감속 명령이 내려간다. "
             "2025년 12월 독일 StVFernLV 시행으로 공공도로 운행의 법적 근거도 마련됐다.",
         size=11.5, line=1.55, first=True)

    concept_diagram(slide, 0.55, 3.95)

    # ---------------- 좌측 하단: 수치 3개 ----------------
    facts = [
        ("6", "오퍼레이터가 동시에 보는 카메라 채널 수"),
        ("64%", "동시 감시 화면이 16개일 때 이상 사건 탐지 정확도\n— 4개일 때는 83% (Tickner & Poulton, 1973)"),
        ("200ms", "이 정도 지연만으로도 차로 유지 노력과\n워크로드가 증가 (Neumeier et al., 2019)"),
    ]
    y = 5.52
    for num, desc in facts:
        rule_line(slide, 0.55, y, 4.1)
        tf = textbox(slide, 0.55, y + 0.10, 0.95, 0.35)
        para(tf, num, size=17, bold=False, color=SIGNAL, font=MONO, first=True)
        tf = textbox(slide, 1.55, y + 0.12, 3.1, 0.5)
        para(tf, desc, size=8.5, color=MUTED, line=1.3, first=True)
        y += 0.62

    # ---------------- 우측: 실촬영 6채널 ----------------
    tf = textbox(slide, 5.25, 0.50, 7.6, 0.4)
    para(tf, "오퍼레이터가 실제로 보는 화면", size=19, bold=True, first=True)
    tf = textbox(slide, 5.25, 0.92, 7.6, 0.3)
    para(tf, "한 시점(0.5초)에 6채널이 동시에 들어온다", size=10.5, color=MUTED, first=True)

    cw, ch, gap = 2.47, 1.39, 0.09
    x0, y0 = 5.25, 1.38
    for i, (cam, ko) in enumerate(GRID):
        cx = x0 + (i % 3) * (cw + gap)
        cy = y0 + (i // 3) * (ch + gap)
        slide.shapes.add_picture(os.path.join(img_dir, f"{cam}.jpg"),
                                 Inches(cx), Inches(cy), Inches(cw), Inches(ch))
        # 캡션 띠
        cap = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx),
                                     Inches(cy + ch - 0.24), Inches(cw), Inches(0.24))
        cap.fill.solid(); cap.fill.fore_color.rgb = RGBColor(0x0A, 0x0E, 0x11)
        cap.fill.transparency = 0.28
        cap.line.fill.background(); cap.shadow.inherit = False
        tfc = cap.text_frame
        tfc.vertical_anchor = MSO_ANCHOR.MIDDLE
        tfc.margin_left = Inches(0.07); tfc.margin_right = Inches(0.07)
        p = tfc.paragraphs[0]
        r = p.add_run(); r.text = cam
        r.font.size = Pt(7.5); r.font.color.rgb = WHITE; r.font.name = MONO
        r = p.add_run(); r.text = "   " + ko
        r.font.size = Pt(7.5); r.font.color.rgb = RGBColor(0xC8, 0xCE, 0xD2); r.font.name = BODY

    # ---------------- 우측 하단: 메시지 + 출처 ----------------
    tf = textbox(slide, 5.25, 4.55, 5.1, 1.5)
    p = tf.paragraphs[0]
    for txt, bold, color in [
        ("전방에는 횡단하는 보행자, 우측에는 이륜차와 군중, 후방에는 접근하는 차량 — ", False, INK),
        ("여러 방향에서 동시에 상황이 벌어지지만 화면은 모두 같은 크기로 균등하게 표시된다.", True, SIGNAL),
        (" 지금 봐야 할 곳을 찾는 일은 온전히 사람의 몫이다.", False, INK),
    ]:
        r = p.add_run(); r.text = txt
        r.font.size = Pt(12); r.font.bold = bold
        r.font.color.rgb = color; r.font.name = BODY
    p.line_spacing = 1.5

    # 연구로 잇는 한 줄 (좌측 강조 레일 + 본문)
    rail = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.25), Inches(5.85),
                                  Pt(2.5), Inches(0.62))
    rail.fill.solid(); rail.fill.fore_color.rgb = ACCENT
    rail.line.fill.background(); rail.shadow.inherit = False
    tf = textbox(slide, 5.48, 5.90, 4.9, 0.6)
    para(tf, "본 연구는 이 6채널 중 지금 주의해야 할 뷰를\n자동으로 골라 강조한다",
         size=12, bold=True, color=ACCENT, line=1.4, first=True)

    tf = textbox(slide, 10.26, 6.35, 2.4, 0.9)
    ttc = f"{min_ttc:.1f}초" if min_ttc >= 0 else "없음"
    para(tf, f"{scene} · boston-seaport", size=8, color=MUTED, font=MONO,
         first=True, align=PP_ALIGN.RIGHT)
    para(tf, f"주석 객체 {n_obj}개 · 최소 TTC {ttc}", size=8, color=MUTED,
         font=MONO, align=PP_ALIGN.RIGHT)
    para(tf, "이미지: nuScenes (CC BY-NC-SA 4.0)", size=8, color=MUTED,
         font=MONO, align=PP_ALIGN.RIGHT)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    prs.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="results/slides")
    ap.add_argument("--out", default="results/slides/원격주행_소개.pptx")
    ap.add_argument("--scene", default="scene-0538")
    ap.add_argument("--n_obj", type=int, default=63)
    ap.add_argument("--min_ttc", type=float, default=2.5)
    args = ap.parse_args()
    path = build(args.img_dir, args.out, args.scene, args.n_obj, args.min_ttc)
    print(f"✅ 슬라이드 생성: {path}  ({os.path.getsize(path)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
