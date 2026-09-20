"""발표용 .pptx 공용 부품.

도형·텍스트를 네이티브 개체로 넣기 위한 얇은 래퍼다. 이미지로 구워 넣으면 발표자가
현장에서 수치 하나를 못 고치므로, 그림 파일 외에는 전부 편집 가능한 개체로 만든다.

색·서체는 연구계획서(docs/research-proposal.html)의 라이트 모드 토큰과 같은 값이다.
"""

from __future__ import annotations

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ---- 색 (계획서 :root 토큰과 동일) ----
INK = RGBColor(0x14, 0x18, 0x1C)
MUTED = RGBColor(0x5A, 0x64, 0x6C)
RULE = RGBColor(0xD8, 0xDE, 0xE2)
SURF = RGBColor(0xFF, 0xFF, 0xFF)
SURF2 = RGBColor(0xEE, 0xF1, 0xF3)
PAPER = RGBColor(0xF4, 0xF6, 0xF7)
ACCENT = RGBColor(0x0D, 0x6B, 0x73)
ACCENT_BG = RGBColor(0xE2, 0xF0, 0xF0)
SIGNAL = RGBColor(0xB8, 0x42, 0x2A)
SIGNAL_BG = RGBColor(0xF7, 0xE7, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

BODY = "맑은 고딕"
MONO = "Consolas"

W, H = 13.333, 7.5          # 16:9
MARGIN = 0.62


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, *, size=12, bold=False, color=INK, font=BODY,
         space_after=0, space_before=0, line=None, first=False,
         align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    if line:
        p.line_spacing = line
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


def rich(tf, parts, *, size=12, line=1.35, space_after=4, first=False,
         align=PP_ALIGN.LEFT):
    """(텍스트, 굵게, 색) 튜플을 한 문단 안에 이어 붙인다 — 문장 중 강조용."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = line
    p.space_after = Pt(space_after)
    for part in parts:
        text, bold, color = (part + (None,) * 3)[:3]
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bool(bold)
        r.font.color.rgb = color or INK
        r.font.name = BODY
    return p


def rule_line(slide, x, y, w, color=RULE, pt=0.75):
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Pt(pt))
    ln.fill.solid()
    ln.fill.fore_color.rgb = color
    ln.line.fill.background()
    ln.shadow.inherit = False
    return ln


def box(slide, x, y, w, h, *, shape=MSO_SHAPE.RECTANGLE, fill=SURF2,
        line=RULE, line_pt=1.0):
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
    sh.text_frame.margin_left = sh.text_frame.margin_right = Inches(0.12)
    sh.text_frame.margin_top = sh.text_frame.margin_bottom = Inches(0.08)
    return sh


def picture(slide, path, x, y, max_w, max_h):
    """상자 (x,y,max_w,max_h) 안에 비율을 지켜 중앙 배치."""
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    return slide.shapes.add_picture(path, Inches(x + (max_w - w) / 2),
                                    Inches(y + (max_h - h) / 2),
                                    Inches(w), Inches(h))


def table(slide, x, y, w, rows, *, col_w, head=True, size=10.5, row_h=0.36,
          head_h=0.30, key_bold=True):
    """단순 표 — python-pptx 기본 표는 스타일이 과해서 직접 그린다."""
    yy = y
    for ri, row in enumerate(rows):
        is_head = head and ri == 0
        h = head_h if is_head else row_h
        xx = x
        if is_head:
            bg = box(slide, x, yy, w, h, fill=SURF2, line=None)
            bg.text_frame.text = ""
        for ci, cell in enumerate(row):
            cw = col_w[ci] * w
            tf = textbox(slide, xx + 0.10, yy + (0.05 if is_head else 0.06),
                         cw - 0.20, h - 0.10,
                         anchor=MSO_ANCHOR.MIDDLE if is_head else MSO_ANCHOR.TOP)
            para(tf, cell, size=size - 1 if is_head else size,
                 bold=is_head or (key_bold and ci == 0 and not is_head),
                 color=MUTED if is_head else INK,
                 font=MONO if is_head else BODY,
                 line=1.25, first=True)
            xx += cw
        yy += h
        if not is_head:
            rule_line(slide, x, yy, w, color=RULE, pt=0.5)
    return yy


def slide_head(slide, kicker, title, *, sub=None):
    """상단 머리 — 번호 kicker + 제목 + 밑줄."""
    tf = textbox(slide, MARGIN, 0.42, 10.0, 0.24)
    para(tf, kicker, size=10.5, color=ACCENT, font=MONO, first=True)
    tf = textbox(slide, MARGIN, 0.66, W - 2 * MARGIN, 0.44)
    para(tf, title, size=25, bold=True, first=True)
    y = 1.20
    if sub:
        tf = textbox(slide, MARGIN, 1.16, W - 2 * MARGIN, 0.30)
        para(tf, sub, size=12.5, color=MUTED, line=1.3, first=True)
        y = 1.52
    rule_line(slide, MARGIN, y, W - 2 * MARGIN, color=RULE, pt=1.0)
    return y + 0.26


def page_num(slide, n):
    tf = textbox(slide, W - MARGIN - 1.0, H - 0.52, 1.0, 0.26)
    para(tf, f"{n:02d}", size=10, color=MUTED, font=MONO, first=True,
         align=PP_ALIGN.RIGHT)


def new_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])      # 빈 레이아웃
    bg = box(s, 0, 0, W, H, fill=WHITE, line=None)
    bg.shadow.inherit = False
    return s
