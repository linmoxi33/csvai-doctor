"""Generate the social preview card.

Kept in the repo rather than run once, because the text in the image is a claim
about behaviour and has to be regenerated whenever that claim changes.

Note on fonts: Consolas and Segoe UI contain no CJK glyphs, so Chinese text
renders as hollow boxes. Microsoft YaHei covers both Latin and CJK, which is why
one family is used throughout.
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
BG = (250, 247, 242)
INK = (26, 24, 22)
BRICK = (166, 61, 41)
MUTED = (120, 112, 104)
RULE = (214, 205, 194)
TERM_BG = (38, 34, 31)
TERM_BAR = (52, 47, 43)

FONT_DIRS = ['C:/Windows/Fonts/', '/usr/share/fonts/truetype/', '']
CJK_CANDIDATES = ['msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'NotoSansCJK-Regular.ttc']


def find_font(names):
    for directory in FONT_DIRS:
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


def load(size, bold=False):
    names = CJK_CANDIDATES if not bold else ['msyhbd.ttc'] + CJK_CANDIDATES
    path = find_font(names)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main(output):
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    display = load(62, bold=True)
    body = load(25)
    small = load(18)
    code = load(20)

    draw.rectangle([0, 0, W, 8], fill=BRICK)
    draw.text((72, 62), 'CSVKIT · OPTIONAL AI LAYER', font=small, fill=MUTED)
    draw.line([72, 98, 300, 98], fill=BRICK, width=2)

    draw.text((72, 122), 'csvai', font=display, fill=INK)
    draw.text((72, 212), '自然语言问 CSV，', font=body, fill=INK)
    draw.text((72, 250), '模型不可用时照样跑。', font=body, fill=INK)

    bx, by, bw, bh = 72, 306, 1136, 224
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=10, fill=TERM_BG)
    draw.rounded_rectangle([bx, by, bx + bw, by + 34], radius=10, fill=TERM_BAR)
    draw.rectangle([bx, by + 24, bx + bw, by + 34], fill=TERM_BAR)
    for index, colour in enumerate([(220, 90, 80), (215, 170, 70), (110, 175, 110)]):
        draw.ellipse([bx + 16 + index * 24, by + 12, bx + 28 + index * 24, by + 24],
                     fill=colour)
    draw.text((bx + 104, by + 7), 'orders.csv', font=small, fill=(170, 162, 152))

    lines = [
        ('$ csvcut --ai "每个城市的订单总金额，从高到低" orders.csv', (228, 221, 211)),
        ('城市,总金额', (140, 200, 150)),
        ('上海,6860.0', (205, 200, 190)),
        ('成都,2118.4', (205, 200, 190)),
        ('重庆,540.5', (205, 200, 190)),
    ]
    y = by + 44
    for text, colour in lines:
        draw.text((bx + 24, y), text, font=code, fill=colour)
        y += 36

    draw.line([72, 578, 1208, 578], fill=RULE, width=1)
    draw.text((72, 594), '$ csvcut --ai "…"   ·   模型 → 受约束的 argv → 真实 csvkit 执行',
              font=small, fill=MUTED)
    draw.text((1010, 592), 'MIT', font=small, fill=BRICK)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    img.save(output, optimize=True)
    print(f'saved {output} ({os.path.getsize(output)} bytes)')
    return 0


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'assets/cover.png'
    sys.exit(main(target))
