#!/usr/bin/env python3
"""Generate the shiftpaper icon SVGs from their geometry.

The icon is a picture frame lying flat on a ground plane, with a mountain
standing up out of it and a sun floating behind. Everything is built from
lines, fillets, one ellipse and one circle, so a change such as the frame
weight is one number here followed by a re-run:

    python3 assets/icon/build.py

Only the standard library is needed. Every SVG in this directory is written
by this script, so edit the geometry here rather than the files.
"""

import math
from dataclasses import dataclass
from pathlib import Path

OUT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Geometry:
    # Frame: centreline of a symmetric trapezoid on the ground plane.
    cx: float
    top_y: float
    bot_y: float
    top_half: float
    bot_half: float
    corner_r: float
    stroke: float
    # Clearance between the mountain's flanks and the frame's back edge.
    gap: float
    # Mountain base: an ellipse on the same plane.
    ex: float
    ey: float
    erx: float
    ery: float
    # Left peak, valley and right peak before rounding, with fillet radii.
    p1: tuple
    p1r: float
    p2: tuple
    p2r: float
    p3: tuple
    p3r: float
    # Sun.
    sx: float
    sy: float
    sr: float
    # When set, the back edge ends square at these x positions instead of
    # parallel to the flanks, so small sizes can land them on pixel edges.
    cut_lx: float | None = None
    cut_rx: float | None = None


# Units are pixels of the original 455x396 concept drawing, which this
# geometry was fitted to. The concept's frame stroke was 14.6.
FULL = Geometry(
    cx=229.07, top_y=179.38, bot_y=353.78, top_half=147.5, bot_half=216.77,
    corner_r=16.31, stroke=18.0, gap=11.25,
    ex=229.0, ey=245.44, erx=124.28, ery=50.0,
    p1=(161.32, 110.28), p1r=21.06,
    p2=(208.72, 200.64), p2r=11.31,
    p3=(271.91, 53.85), p3r=25.4,
    sx=346.3, sy=73.74, sr=32.3,
)

# Redrawn on a 16px grid for 16-24px: one-pixel strokes on whole rows, a
# narrower mountain so the back edge still shows either side of it, square
# ends on pixel edges and a larger sun.
SMALL = Geometry(
    cx=8.0, top_y=6.5, bot_y=13.5, top_half=5.5, bot_half=7.5,
    corner_r=0.5, stroke=1.0, gap=1.0,
    ex=8.0, ey=9.9, erx=3.8, ery=1.9,
    p1=(5.3, 3.4), p1r=0.6,
    p2=(7.4, 6.4), p2r=0.3,
    p3=(9.6, 0.2), p3r=0.8,
    sx=13.25, sy=2.75, sr=1.75,
    cut_lx=4.0, cut_rx=12.0,
)

# GitHub's default text colours, so the README icon matches the heading.
README_LIGHT = "#1f2328"
README_DARK = "#f0f6fc"


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(a, k):
    return (a[0] * k, a[1] * k)


def unit(a):
    n = math.hypot(*a)
    return (a[0] / n, a[1] / n)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def fillet(prev, corner, nxt, r):
    """Tangent points, centre and sweep flag for rounding a corner by r."""
    u = unit(sub(prev, corner))
    v = unit(sub(nxt, corner))
    half = math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))) / 2
    t = r / math.tan(half)
    centre = add(corner, mul(unit(add(u, v)), r / math.sin(half)))
    # A right turn on screen (y down) is clockwise, SVG's sweep-flag 1.
    sweep = 1 if cross(sub(corner, prev), sub(nxt, corner)) > 0 else 0
    return add(corner, mul(u, t)), add(corner, mul(v, t)), centre, sweep


def ellipse_tangent(pt, g, side):
    """Where a line from pt touches the base ellipse; side -1 left, +1 right."""
    q = ((pt[0] - g.ex) / g.erx, (pt[1] - g.ey) / g.ery)
    base = math.atan2(q[1], q[0])
    off = math.acos(1 / math.hypot(*q))
    points = [(g.ex + g.erx * math.cos(a), g.ey + g.ery * math.sin(a)) for a in (base + off, base - off)]
    return max(points, key=lambda p: p[0] * side)


def x_at(a, b, y):
    return a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1])


def trapezoid(g, off):
    """Frame corners TL, TR, BR, BL with every edge moved outward by off."""
    pts = [
        (g.cx - g.top_half, g.top_y),
        (g.cx + g.top_half, g.top_y),
        (g.cx + g.bot_half, g.bot_y),
        (g.cx - g.bot_half, g.bot_y),
    ]
    lines = []
    for i in range(4):
        d = unit(sub(pts[(i + 1) % 4], pts[i]))
        lines.append((add(pts[i], mul((d[1], -d[0]), off)), d))
    corners = []
    for i in range(4):
        (a0, d0), (a1, d1) = lines[i - 1], lines[i]
        corners.append(add(a0, mul(d0, cross(sub(a1, a0), d1) / cross(d0, d1))))
    return corners


def flanks(g):
    tl = ellipse_tangent(g.p1, g, -1)
    tr = ellipse_tangent(g.p3, g, +1)
    return tl, tr


class Pen:
    """Writes path data, mapping geometry units into the output viewBox."""

    def __init__(self, scale=1.0, dx=0.0, dy=0.0):
        self.scale, self.dx, self.dy = scale, dx, dy
        self.parts = []

    def num(self, v):
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return "0" if s in ("", "-0") else s

    def pt(self, p):
        return f"{self.num(p[0] * self.scale + self.dx)} {self.num(p[1] * self.scale + self.dy)}"

    def move(self, p):
        self.parts.append(f"M{self.pt(p)}")

    def line(self, p):
        self.parts.append(f"L{self.pt(p)}")

    def arc(self, rx, ry, large, sweep, p):
        r = f"{self.num(rx * self.scale)} {self.num(ry * self.scale)}"
        self.parts.append(f"A{r} 0 {large} {sweep} {self.pt(p)}")

    def close(self):
        self.parts.append("Z")

    def corner(self, prev, corner, nxt, r):
        if r <= 0:
            self.line(corner)
            return
        a, b, _, sweep = fillet(prev, corner, nxt, r)
        self.line(a)
        self.arc(r, r, 0, sweep, b)

    def d(self):
        return "".join(self.parts)


def draw_frame(g, pen):
    """The frame as one filled outline, open where the mountain passes."""
    h = g.stroke / 2
    outer, inner = trapezoid(g, h), trapezoid(g, -h)
    ro, ri = g.corner_r + h, max(g.corner_r - h, 0.0)
    yo, yi = outer[0][1], inner[0][1]

    tl, tr = flanks(g)
    nl = unit(sub(tl, g.p1))
    nr = unit(sub(tr, g.p3))
    left = [add(p, mul((-nl[1], nl[0]), g.gap)) for p in (g.p1, tl)]
    right = [add(p, mul((nr[1], -nr[0]), g.gap)) for p in (g.p3, tr)]
    lx_o, lx_i = (g.cut_lx, g.cut_lx) if g.cut_lx is not None else (x_at(*left, yo), x_at(*left, yi))
    rx_o, rx_i = (g.cut_rx, g.cut_rx) if g.cut_rx is not None else (x_at(*right, yo), x_at(*right, yi))

    # Outside edge anticlockwise from the left cut, inside edge back again.
    route = [((lx_o, yo), outer, [0, 3, 2, 1], ro, (rx_o, yo)), ((rx_i, yi), inner, [1, 2, 3, 0], ri, (lx_i, yi))]
    pen.move(route[0][0])
    for start, poly, order, r, end in route:
        if start != route[0][0]:
            pen.line(start)
        for k, i in enumerate(order):
            prev = poly[order[k - 1]] if k else start
            nxt = poly[order[k + 1]] if k < 3 else end
            pen.corner(prev, poly[i], nxt, r)
        pen.line(end)
    pen.close()


def draw_mountain(g, pen):
    tl, tr = flanks(g)
    pen.move(tl)
    pen.corner(tl, g.p1, g.p2, g.p1r)
    pen.corner(g.p1, g.p2, g.p3, g.p2r)
    pen.corner(g.p2, g.p3, tr, g.p3r)
    pen.line(tr)
    # The flanks touch the ellipse above its widest point, so the visible
    # base is the larger arc.
    pen.arc(g.erx, g.ery, 1, 1, tl)
    pen.close()


def draw_sun(g, pen):
    a, b = (g.sx - g.sr, g.sy), (g.sx + g.sr, g.sy)
    pen.move(a)
    pen.arc(g.sr, g.sr, 1, 0, b)
    pen.arc(g.sr, g.sr, 1, 0, a)
    pen.close()


def path_d(g, scale=1.0, dx=0.0, dy=0.0):
    pen = Pen(scale, dx, dy)
    draw_frame(g, pen)
    draw_mountain(g, pen)
    draw_sun(g, pen)
    return pen.d()


def bounds(g):
    """Exact ink bounds: frame bottom corners, frame base, peak and sun."""
    h = g.stroke / 2
    outer = trapezoid(g, h)
    ro = g.corner_r + h
    _, _, bl, _ = fillet(outer[0], outer[3], outer[2], ro)
    _, _, br, _ = fillet(outer[3], outer[2], outer[1], ro)
    tl, tr = flanks(g)
    _, _, peak, _ = fillet(g.p2, g.p3, tr, g.p3r)
    x0 = bl[0] - ro
    x1 = max(br[0] + ro, g.sx + g.sr)
    y0 = min(peak[1] - g.p3r, g.sy - g.sr)
    y1 = g.bot_y + h
    return x0, y0, x1, y1


def fit(g, size, width):
    """Scale and offset that centre the ink, `width` units wide, in a square."""
    x0, y0, x1, y1 = bounds(g)
    s = width / (x1 - x0)
    return s, (size - (x1 - x0) * s) / 2 - x0 * s, (size - (y1 - y0) * s) / 2 - y0 * s


def svg(view, size, body, title="shiftpaper", extra=""):
    w, h = size
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{view}"{extra}>'
        f"<title>{title}</title>{body}</svg>\n"
    )


def glyph(g, size, width, fill="currentColor"):
    s, dx, dy = fit(g, size, width)
    return svg(f"0 0 {size} {size}", (size, size), f'<path fill="{fill}" d="{path_d(g, s, dx, dy)}"/>')


def readme(g, fill):
    """Cropped to the ink so the README can size it by width alone."""
    x0, y0, x1, y1 = bounds(g)
    s = 120 / (x1 - x0)
    w, h = 120, math.ceil((y1 - y0) * s)
    dy = (h - (y1 - y0) * s) / 2 - y0 * s
    return svg(f"0 0 {w} {h}", (w, h), f'<path fill="{fill}" d="{path_d(g, s, -x0 * s, dy)}"/>')


def app_icon(g):
    """Light glyph on a dark tile, so it shows on light and dark desktops."""
    s, dx, dy = fit(g, 128, 84)
    body = (
        '<defs><linearGradient id="tile" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#2c3039"/><stop offset="1" stop-color="#16181d"/>'
        "</linearGradient></defs>"
        '<rect x="8" y="8" width="112" height="112" rx="24" fill="url(#tile)"/>'
        f'<path fill="#f2f3f5" d="{path_d(g, s, dx, dy + 1)}"/>'
    )
    return svg("0 0 128 128", (128, 128), body)


def symbolic(g):
    """Small glyph using the Breeze colour scheme class; GTK recolours it too."""
    body = (
        '<style id="current-color-scheme">.ColorScheme-Text{color:#232629}</style>'
        f'<path class="ColorScheme-Text" fill="currentColor" d="{path_d(g)}"/>'
    )
    return svg("0 0 16 16", (16, 16), body)


def main():
    files = {
        "shiftpaper.svg": glyph(FULL, 128, 116),
        "shiftpaper-small.svg": svg("0 0 16 16", (16, 16), f'<path fill="currentColor" d="{path_d(SMALL)}"/>'),
        "readme-light.svg": readme(FULL, README_LIGHT),
        "readme-dark.svg": readme(FULL, README_DARK),
        "hicolor/scalable/apps/shiftpaper.svg": app_icon(FULL),
        "hicolor/symbolic/apps/shiftpaper-symbolic.svg": symbolic(SMALL),
    }
    for name, text in files.items():
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
