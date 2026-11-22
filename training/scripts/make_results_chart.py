"""Draw docs/img/results.svg from the experiment summaries (no plotting library needed).

Run from `training`:  ../backend/.venv/Scripts/python scripts/make_results_chart.py
Each dot is the mean over seeds of the test-set perplexity ratio vs the same-source random baseline
(paired per seed); the whisker is the min-max over seeds. Lower than 1.0 = better than random.
Arms with one seed (medical r1-mix, legal and code `curated`) have no whisker.
"""
import json
from pathlib import Path

ROOT = Path("data/experiments")
PLANS = [("medical", "medical-160m", "summary-recipes", "medical"),
         ("legal", "legal-160m", "summary", "legal"),
         ("code", "code-160m", "summary", "code")]
ARMS = [("curated", {"medical": "curated"}, "#a23b3b"),
        ("curated-mix", {"medical": "r1-mix"}, "#2a6fb0"),
        ("loose-mix", {"medical": "r2-loose-mix"}, "#5b9bd5"),
        ("control (unfiltered mix)", {"medical": "control-random-mix", "legal": "control-random-mix",
                                      "code": "control-random-mix"}, "#7a7a7a"),
        ("wiki-mix (raw Wikipedia)", {}, "#c28a1c")]


def ratios(plan, summary, arm, task):
    s = json.loads((ROOT / plan / f"{summary}.json").read_text())
    vs = s["variants"]["pretrained"]["vs_reference"].get(arm)
    if not vs:
        return None
    return [c[task]["ratio_b_over_a"] for c in vs.values() if task in c]


def main():
    W, H, left, pw, gap, rowh = 1020, 470, 150, 220, 70, 175
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W/2}" y="22" text-anchor="middle" font-size="15" font-weight="600">Test perplexity vs same-source random baseline '
           f'(Pythia-160M, 20M tokens; lower is better; dot = mean of 3 seeds, whisker = min-max)</text>']
    rows = [("domain text", None, (0.95, 1.05), [0.96, 0.98, 1.0, 1.02, 1.04]),
            ("WikiText (general)", "wikitext", (0.89, 1.01), [0.90, 0.92, 0.94, 0.96, 0.98, 1.0])]
    for r, (rname, rtask, (lo, hi), ticks) in enumerate(rows):
        for c, (dom, plan, summ, dtask) in enumerate(PLANS):
            x0 = left + c * (pw + gap) - (0 if c else 0)
            x0 = 150 + c * (pw + gap)
            y0 = 55 + r * rowh
            task = dtask if rtask is None else rtask
            out.append(f'<text x="{x0 + pw/2}" y="{y0 - 8}" text-anchor="middle" font-weight="600">{dom}: {task} ppl ratio</text>')
            def X(v): return x0 + (v - lo) / (hi - lo) * pw
            for t in ticks:
                out.append(f'<line x1="{X(t):.1f}" y1="{y0}" x2="{X(t):.1f}" y2="{y0 + 125}" stroke="#eee"/>')
                out.append(f'<text x="{X(t):.1f}" y="{y0 + 140}" text-anchor="middle" fill="#666">{t:.2f}</text>')
            out.append(f'<line x1="{X(1.0):.1f}" y1="{y0}" x2="{X(1.0):.1f}" y2="{y0 + 125}" stroke="#333" stroke-dasharray="4 3"/>')
            arms = [a for a in ARMS if not (rtask and a[0] == "curated")]
            for i, (label, amap, color) in enumerate(arms):
                arm = amap.get(dom, label.split(" ")[0] if label != "wiki-mix (raw Wikipedia)" else "wiki-mix")
                if label == "curated-mix" and dom != "medical":
                    arm = "curated-mix"
                if label == "loose-mix" and dom != "medical":
                    arm = "loose-mix"
                if label.startswith("control"):
                    arm = "control-random-mix"
                v = ratios(plan, summ, arm, task)
                y = y0 + 12 + i * 24
                if c == 0:
                    out.append(f'<text x="{x0 - 8}" y="{y + 4}" text-anchor="end">{label}</text>')
                if not v:
                    continue
                m = sum(v) / len(v)
                if min(v) < lo or max(v) > hi:
                    out.append(f'<text x="{x0 + pw + 4}" y="{y + 4}" fill="{color}" font-size="11">{m:.2f} (off scale)</text>')
                    continue
                if len(v) > 1:
                    out.append(f'<line x1="{X(min(v)):.1f}" y1="{y}" x2="{X(max(v)):.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>')
                out.append(f'<circle cx="{X(m):.1f}" cy="{y}" r="5" fill="{color}"/>')
                out.append(f'<text x="{X(m):.1f}" y="{y - 8}" text-anchor="middle" font-size="10" fill="#444">{m:.3f}</text>')
    out.append(f'<text x="{W/2}" y="{H - 8}" text-anchor="middle" fill="#666">medical curated-mix = r1-mix and loose-mix = r2-loose-mix (1 seed: medical r1-mix, legal and code curated); '
               f'curated is omitted from the WikiText row (1.4-2.7x, off scale)</text>')
    out.append("</svg>")
    Path("../docs/img/results.svg").write_text("\n".join(out), encoding="utf-8")
    print("wrote ../docs/img/results.svg")


main()
