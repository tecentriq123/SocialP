"""Small SVG plotting helper for the stats site.

All colors come from CSS classes so the figures follow the page theme:
  series stroke  : class "s1".."s4"  (stroke = var(--s1)...)
  series fill    : class "f1".."f4"  (fill = var(--s1)...)
  soft fill      : class "a1".."a4"  (fill = series color, low opacity)
  muted ink      : class "mute"      (text / reference lines)
Text inherits the page font; sizes are in viewBox units.
"""
import math
from html import escape

FONT = 13  # tick / label font size in viewBox units


def fmt(v, nd=None):
    if nd is not None:
        s = f"{v:.{nd}f}"
    else:
        s = f"{v:.6g}"
    if "." in s and nd is None:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    v = start
    while v <= hi + 1e-9 * step:
        ticks.append(round(v, 10))
        v += step
    return ticks


class Plot:
    def __init__(self, xlim, ylim, w=600, h=340, ml=62, mr=24, mt=22, mb=54,
                 xlabel="", ylabel="", xticks=None, yticks=None,
                 xtickfmt=None, ytickfmt=None, xticklabels=None,
                 ygrid=True, xgrid=False, show_yaxis=True, show_xaxis=True):
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.w, self.h = w, h
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xticks = nice_ticks(*xlim) if xticks is None else xticks
        self.yticks = nice_ticks(*ylim) if yticks is None else yticks
        self.xtickfmt = xtickfmt or (lambda v: fmt(v))
        self.ytickfmt = ytickfmt or (lambda v: fmt(v))
        self.xticklabels = xticklabels
        self.ygrid, self.xgrid = ygrid, xgrid
        self.show_yaxis, self.show_xaxis = show_yaxis, show_xaxis
        self.els = []
        self.top = []  # drawn after everything (labels)

    # scales
    def sx(self, x):
        return self.ml + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.ml - self.mr)

    def sy(self, y):
        return self.h - self.mb - (y - self.y0) / (self.y1 - self.y0) * (self.h - self.mt - self.mb)

    def _d(self, xs, ys):
        pts = [f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in zip(xs, ys)]
        return "M" + " L".join(pts)

    # marks
    def line(self, xs, ys, s=1, dash=False, w=2, extra=""):
        da = ' stroke-dasharray="6 5"' if dash else ""
        self.els.append(f'<path d="{self._d(xs, ys)}" class="ln s{s}" stroke-width="{w}"{da} fill="none"{extra}/>')

    def step(self, xs, ys, s=1, dash=False, w=2):
        """post-step (Kaplan-Meier style): value ys[i] holds from xs[i] to xs[i+1]"""
        px, py = [xs[0]], [ys[0]]
        for i in range(1, len(xs)):
            px += [xs[i], xs[i]]
            py += [ys[i - 1], ys[i]]
        self.line(px, py, s=s, dash=dash, w=w)

    def fill_between(self, xs, y0s, y1s, s=1, cls=None):
        top = [f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in zip(xs, y1s)]
        bot = [f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in zip(reversed(xs), reversed(y0s))]
        c = cls or f"a{s}"
        self.els.append(f'<polygon points="{" ".join(top + bot)}" class="{c}"/>')

    def points(self, xs, ys, s=1, r=4, hollow=False):
        cls = f"pth s{s}" if hollow else f"pt f{s}"
        for x, y in zip(xs, ys):
            self.els.append(f'<circle cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="{r}" class="{cls}"/>')

    def ticks_marks(self, xs, ys, s=1, size=5):
        """small vertical ticks (censoring marks)"""
        for x, y in zip(xs, ys):
            X, Y = self.sx(x), self.sy(y)
            self.els.append(f'<line x1="{X:.1f}" y1="{Y - size:.1f}" x2="{X:.1f}" y2="{Y + size:.1f}" class="ln s{s}" stroke-width="1.5"/>')

    def hist(self, edges, counts, s=1, gap=1.0, cls=None):
        c = cls or f"f{s}"
        for i, cnt in enumerate(counts):
            xa, xb = self.sx(edges[i]), self.sx(edges[i + 1])
            ya, yb = self.sy(cnt), self.sy(self.y0)
            wdt = max(xb - xa - gap, 0.5)
            self.els.append(f'<rect x="{xa + gap / 2:.1f}" y="{ya:.1f}" width="{wdt:.1f}" height="{max(yb - ya, 0):.1f}" class="{c}"/>')

    def bars(self, xs, heights, width, s=1, rounded=True, cls=None):
        c = cls or f"f{s}"
        for x, hgt in zip(xs, heights):
            xa, xb = self.sx(x - width / 2), self.sx(x + width / 2)
            ya, yb = self.sy(hgt), self.sy(0)
            top, bottom = min(ya, yb), max(ya, yb)
            wd, ht = xb - xa, bottom - top
            if rounded and ht > 4:
                r = min(4, wd / 2)
                d = (f"M{xa:.1f},{bottom:.1f} L{xa:.1f},{top + r:.1f} Q{xa:.1f},{top:.1f} {xa + r:.1f},{top:.1f} "
                     f"L{xb - r:.1f},{top:.1f} Q{xb:.1f},{top:.1f} {xb:.1f},{top + r:.1f} L{xb:.1f},{bottom:.1f} Z")
                self.els.append(f'<path d="{d}" class="{c}"/>')
            else:
                self.els.append(f'<rect x="{xa:.1f}" y="{top:.1f}" width="{wd:.1f}" height="{ht:.1f}" class="{c}"/>')

    def vline(self, x, cls="ref", dash=True, y0=None, y1=None, w=1.2):
        da = ' stroke-dasharray="4 4"' if dash else ""
        ya = self.sy(self.y0 if y0 is None else y0)
        yb = self.sy(self.y1 if y1 is None else y1)
        self.els.append(f'<line x1="{self.sx(x):.1f}" y1="{ya:.1f}" x2="{self.sx(x):.1f}" y2="{yb:.1f}" class="{cls}" stroke-width="{w}"{da}/>')

    def hline(self, y, cls="ref", dash=True, x0=None, x1=None, w=1.2):
        da = ' stroke-dasharray="4 4"' if dash else ""
        xa = self.sx(self.x0 if x0 is None else x0)
        xb = self.sx(self.x1 if x1 is None else x1)
        self.els.append(f'<line x1="{xa:.1f}" y1="{self.sy(y):.1f}" x2="{xb:.1f}" y2="{self.sy(y):.1f}" class="{cls}" stroke-width="{w}"{da}/>')

    def seg(self, xa, ya, xb, yb, cls="ref", w=1.2, dash=False):
        da = ' stroke-dasharray="4 4"' if dash else ""
        self.els.append(f'<line x1="{self.sx(xa):.1f}" y1="{self.sy(ya):.1f}" x2="{self.sx(xb):.1f}" y2="{self.sy(yb):.1f}" class="{cls}" stroke-width="{w}"{da}/>')

    def text(self, x, y, s, anchor="start", cls="lbl", dx=0, dy=0, size=None, weight=None):
        fs = f' font-size="{size}"' if size else ""
        fw = f' font-weight="{weight}"' if weight else ""
        self.top.append(f'<text x="{self.sx(x) + dx:.1f}" y="{self.sy(y) + dy:.1f}" text-anchor="{anchor}" class="{cls}"{fs}{fw}>{escape(s)}</text>')

    def text_px(self, X, Y, s, anchor="start", cls="lbl", size=None, weight=None):
        fs = f' font-size="{size}"' if size else ""
        fw = f' font-weight="{weight}"' if weight else ""
        self.top.append(f'<text x="{X:.1f}" y="{Y:.1f}" text-anchor="{anchor}" class="{cls}"{fs}{fw}>{escape(s)}</text>')

    def legend(self, items, X=None, Y=None, gap=18):
        """items: list of (label, series, kind) kind in line|dash|box|dot. Placed top-left of plot area by default."""
        X = self.ml + 12 if X is None else X
        Y = self.mt + 10 if Y is None else Y
        for i, (label, s, kind) in enumerate(items):
            y = Y + i * gap
            if kind in ("line", "dash"):
                da = ' stroke-dasharray="6 4"' if kind == "dash" else ""
                self.top.append(f'<line x1="{X:.1f}" y1="{y:.1f}" x2="{X + 22:.1f}" y2="{y:.1f}" class="ln s{s}" stroke-width="2.4"{da}/>')
            elif kind == "box":
                self.top.append(f'<rect x="{X + 5:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2" class="f{s}"/>')
            elif kind == "soft":
                self.top.append(f'<rect x="{X + 5:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2" class="a{s} s{s}" stroke-width="1"/>')
            elif kind == "dot":
                self.top.append(f'<circle cx="{X + 11:.1f}" cy="{y:.1f}" r="4.5" class="f{s}"/>')
            self.top.append(f'<text x="{X + 30:.1f}" y="{y + 4.5:.1f}" class="lbl">{escape(label)}</text>')

    # output
    def svg(self, aria="", extra_defs=""):
        o = []
        L, R = self.ml, self.w - self.mr
        T, B = self.mt, self.h - self.mb
        if self.ygrid:
            for t in self.yticks:
                if self.y0 - 1e-9 <= t <= self.y1 + 1e-9:
                    y = self.sy(t)
                    o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" class="grid"/>')
        if self.xgrid:
            for t in self.xticks:
                if self.x0 - 1e-9 <= t <= self.x1 + 1e-9:
                    x = self.sx(t)
                    o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{B}" class="grid"/>')
        if self.show_xaxis:
            o.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" class="axis"/>')
            labels = self.xticklabels or [(t, self.xtickfmt(t)) for t in self.xticks]
            for t, lab in labels:
                if self.x0 - 1e-9 <= t <= self.x1 + 1e-9:
                    x = self.sx(t)
                    o.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B + 4}" class="axis"/>')
                    o.append(f'<text x="{x:.1f}" y="{B + 18}" text-anchor="middle" class="tick">{escape(str(lab))}</text>')
            if self.xlabel:
                o.append(f'<text x="{(L + R) / 2:.1f}" y="{self.h - 10}" text-anchor="middle" class="axlab">{escape(self.xlabel)}</text>')
        if self.show_yaxis:
            for t in self.yticks:
                if self.y0 - 1e-9 <= t <= self.y1 + 1e-9:
                    y = self.sy(t)
                    o.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{escape(self.ytickfmt(t))}</text>')
            if self.ylabel:
                cy = (T + B) / 2
                o.append(f'<text x="16" y="{cy:.1f}" text-anchor="middle" transform="rotate(-90 16 {cy:.1f})" class="axlab">{escape(self.ylabel)}</text>')
        body = "".join(o) + "".join(self.els) + "".join(self.top)
        return (f'<svg viewBox="0 0 {self.w} {self.h}" class="viz" role="img" aria-label="{escape(aria)}" '
                f'xmlns="http://www.w3.org/2000/svg">{extra_defs}{body}</svg>')


def figure(svgs, caption="", cols=1, note=""):
    """Wrap one or more svg strings into a <figure>. cols>1 lays panels in a grid."""
    if isinstance(svgs, str):
        svgs = [svgs]
    cls = "fig" if cols == 1 else f"fig fig-grid g{cols}"
    inner = "".join(f'<div class="fig-panel">{s}</div>' for s in svgs)
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    nt = f'<p class="fig-note">{note}</p>' if note else ""
    return f'<figure class="{cls}"><div class="fig-body">{inner}</div>{cap}{nt}</figure>'


def panel_title(p, s):
    p.top.append(f'<text x="{p.ml}" y="{p.mt - 6}" class="ptitle">{escape(s)}</text>')


def forest(rows, xlim, ref=1.0, log=True, w=640, row_h=30, label_w=210, est_w=150,
           xlabel="Hazard ratio (95% CI)", left_note="", right_note="", xticks=None, header=None):
    """rows: list of dict(label, est, lo, hi, bold=False, header=False, s=1, size=None)
    Draws a forest plot: label column | plot | 'est (lo-hi)' column."""
    tf = (lambda v: math.log(v)) if log else (lambda v: v)
    top_pad = 34 if header else 12
    h = top_pad + row_h * len(rows) + 58
    X0, X1 = label_w, w - est_w
    lo_t, hi_t = tf(xlim[0]), tf(xlim[1])

    def sx(v):
        return X0 + (tf(v) - lo_t) / (hi_t - lo_t) * (X1 - X0)

    o = []
    if header:
        o.append(f'<text x="4" y="18" class="axlab">{escape(header[0])}</text>')
        o.append(f'<text x="{w - 4}" y="18" text-anchor="end" class="axlab">{escape(header[1])}</text>')
    yb = top_pad + row_h * len(rows) + 6
    ticks = xticks or ([0.25, 0.5, 1, 2, 4] if log else nice_ticks(*xlim))
    for t in ticks:
        if xlim[0] <= t <= xlim[1]:
            x = sx(t)
            o.append(f'<line x1="{x:.1f}" y1="{top_pad - 4}" x2="{x:.1f}" y2="{yb}" class="grid"/>')
            o.append(f'<text x="{x:.1f}" y="{yb + 18}" text-anchor="middle" class="tick">{fmt(t)}</text>')
    o.append(f'<line x1="{X0}" y1="{yb}" x2="{X1}" y2="{yb}" class="axis"/>')
    xr = sx(ref)
    o.append(f'<line x1="{xr:.1f}" y1="{top_pad - 4}" x2="{xr:.1f}" y2="{yb}" class="ref" stroke-width="1.4"/>')
    o.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{h - 10}" text-anchor="middle" class="axlab">{escape(xlabel)}</text>')
    if left_note:
        o.append(f'<text x="{xr - 8:.1f}" y="{yb + 36}" text-anchor="end" class="lbl small">{escape(left_note)}</text>')
    if right_note:
        o.append(f'<text x="{xr + 8:.1f}" y="{yb + 36}" class="lbl small">{escape(right_note)}</text>')
    for i, r in enumerate(rows):
        y = top_pad + row_h * i + row_h / 2
        wcls = ' font-weight="600"' if r.get("bold") or r.get("header") else ""
        indent = 4 if (r.get("header") or r.get("bold") or not r.get("indent", False)) else 18
        o.append(f'<text x="{indent}" y="{y + 4.5:.1f}" class="lbl"{wcls}>{escape(r["label"])}</text>')
        if r.get("header") or r.get("est") is None:
            continue
        s = r.get("s", 1)
        a, b = sx(max(r["lo"], xlim[0])), sx(min(r["hi"], xlim[1]))
        o.append(f'<line x1="{a:.1f}" y1="{y:.1f}" x2="{b:.1f}" y2="{y:.1f}" class="ln s{s}" stroke-width="2"/>')
        xe = sx(r["est"])
        if r.get("diamond"):
            dh = 8
            o.append(f'<polygon points="{a:.1f},{y:.1f} {xe:.1f},{y - dh:.1f} {b:.1f},{y:.1f} {xe:.1f},{y + dh:.1f}" class="f{s}"/>')
        else:
            sz = r.get("size", 5)
            o.append(f'<rect x="{xe - sz:.1f}" y="{y - sz:.1f}" width="{2 * sz:.1f}" height="{2 * sz:.1f}" class="f{s}"/>')
        nd = r.get("nd", 2)
        o.append(f'<text x="{w - 4}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num"{wcls}>'
                 f'{r["est"]:.{nd}f} ({r["lo"]:.{nd}f}–{r["hi"]:.{nd}f})</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="viz" role="img" aria-label="forest plot" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')
