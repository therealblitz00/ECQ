"""Phase 6: delivery. One command line for the batch outputs and a static HTML report a buyer can open.

    python src/phase6_deliver.py query 167718_BU     # similar products, why, and the sales / price estimates
    python src/phase6_deliver.py report              # -> outputs/phase6/report.html (self-contained, images inline)
    python src/phase6_deliver.py brief               # -> outputs/phase6/brief_example.md + .jpg: the brief's example

Reads only the outputs of the earlier phases (Phase 3 neighbours, Phase 5 explanations, Phase 5b estimates); it
computes nothing new, so what the report shows is exactly what the batch files contain.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
from config import IMG_DIR, KEY, PROCESSED_DIR, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase6"
BRIEF_QUERY, BRIEF_ANSWER = "167718_BU", ["169597_FU", "169607_IG", "169607_PE", "177781_LB"]  # brief, slide 6
PER_CATEGORY = 2
SHARE_LABEL = {"share_image": "photo", "share_categorical": "attributes", "share_text": "description",
               "share_composition": "materials", "share_palette": "photo colours", "share_numeric": "price & launch"}


def load() -> dict:
    out = {"items": phase1_clean.load_items().set_index(KEY),
           "nb": pd.read_parquet(PROCESSED_DIR / "neighbours.parquet")}
    for name, file in [("ex", "explanations.parquet"), ("fc", "forecast.parquet")]:
        path = PROCESSED_DIR / file
        out[name] = pd.read_parquet(path) if path.exists() else None
    return out


def brief_check(d: dict) -> pd.DataFrame:
    """Where the engine ranks the brief's own four answers for its example product."""
    from phase3_engine import load_blocks, load_weights, neighbours
    items = d["items"].reset_index()
    nb = neighbours(items, load_blocks(items), load_weights(), k=len(items), queries=[BRIEF_QUERY])
    pos = nb.set_index("candidate")["rank"]
    return pd.DataFrame({"brief answer": BRIEF_ANSWER, "engine rank": [int(pos.get(k, -1)) for k in BRIEF_ANSWER],
                         "out of": len(nb)})


def cmd_query(args: argparse.Namespace) -> None:
    d = load()
    it = d["items"]
    if args.key not in it.index:
        raise SystemExit(f"unknown product {args.key}")
    r = it.loc[args.key]
    print(f"{args.key}: {r['PROD_DES_BASE']} | {r['CAT_DES_EN']} › {r['GFA_DES_EN']} › {r['GFS_DES_EN']} | "
          f"{r['CLR_DES']} | {r['PRICE_BASE_W_VAT']:.2f} €")
    if d["ex"] is not None:
        for e in d["ex"][d["ex"]["query"] == args.key].head(args.k).itertuples():
            print(f"  {e.rank}. {e.candidate}  score {e.score:.3f}  {e.summary}")
    else:
        for e in d["nb"][d["nb"]["query"] == args.key].head(args.k).itertuples():
            print(f"  {e.rank}. {e.candidate}  score {e.score:.3f}")
    if d["fc"] is not None:
        f = d["fc"].set_index(KEY).loc[args.key]
        sold = "unknown" if pd.isna(r["SALES_QTY"]) else f"{int(r['SALES_QTY']):,}"
        print(f"  sales: estimate {f['sales_estimate']:,.0f} whole-life units (80% range {f['sales_low']:,.0f} to "
              f"{f['sales_high']:,.0f}; {f['sales_estimate_kind']}); sold {sold}; similar past products sold "
              f"~{f['sales_analogue']:,.0f} ({f['sales_analogue_keys']}); {f['sales_reasons']}")
        print(f"  price: estimate {f['price_estimate']:.2f} € (nearest list price {f['price_point_estimate']:.2f} €, 80% range "
              f"{f['price_low']:.2f} to {f['price_high']:.2f} €; {f['price_estimate_kind']}); list "
              f"{r['PRICE_BASE_W_VAT']:.2f} €; {f['price_reasons']}")


# --------------------------------------------------------------------------- HTML report
def thumb(file, size: int = 150) -> str:
    if not isinstance(file, str):
        return '<div class="noimg">no trusted photo</div>'
    from PIL import Image
    try:
        im = Image.open(IMG_DIR / file).convert("RGB")
    except OSError:
        return '<div class="noimg">photo unreadable</div>'
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return f'<img alt="" src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def card(d: dict, key: str, extra: str = "") -> str:
    r = d["items"].loc[key]
    photo = r["img_file"] if bool(r.get("img_trusted", False)) else None
    return (f'<div class="card">{thumb(photo)}<b>{html.escape(key)}</b>'
            f'<span>{html.escape(str(r["GFS_DES_EN"]))} · {html.escape(str(r["CLR_DES"]))}</span>'
            f'<span>{r["PRICE_BASE_W_VAT"]:.2f} €</span>{extra}</div>')


def shares_bar(e) -> str:
    parts = [(SHARE_LABEL[c], float(getattr(e, c))) for c in SHARE_LABEL if getattr(e, c, 0) > 0]
    segs = "".join(f'<i class="s{n}" style="width:{100 * v:.1f}%" title="{lab} {v:.0%}"></i>'
                   for n, (lab, v) in enumerate(parts))
    legend = " · ".join(f"{lab} {v:.0%}" for lab, v in sorted(parts, key=lambda kv: -kv[1])[:3])
    return f'<div class="bar">{segs}</div><small>{html.escape(legend)}</small>'


def query_block(d: dict, q: str) -> str:
    ex = d["ex"][d["ex"]["query"] == q].head(5)
    cards = []
    for e in ex.itertuples():
        diffs = json.loads(e.differences)
        cards.append(card(d, e.candidate, f'<small>score {e.score:.3f}</small>{shares_bar(e)}'
                                          f'<p>{html.escape(e.summary.split(". ", 1)[-1])}</p>'
                                          + (f'<p class="diff">differs: {html.escape("; ".join(diffs[:2]))}</p>'
                                             if diffs else "")))
    fc = ""
    if d["fc"] is not None:
        f = d["fc"].set_index(KEY).loc[q]
        r = d["items"].loc[q]
        sold = "no sales data" if pd.isna(r["SALES_QTY"]) else f"sold {int(r['SALES_QTY']):,}"
        fc = (f'<p class="fc"><b>Sales estimate</b> {f["sales_estimate"]:,.0f} whole-life units (80% range '
              f'{f["sales_low"]:,.0f} to {f["sales_high"]:,.0f}), {html.escape(f["sales_estimate_kind"])} ({sold}; '
              f'similar past products sold about {f["sales_analogue"]:,.0f}). Main effects: '
              f'{html.escape(f["sales_reasons"])}.<br>'
              f'<b>Price estimate</b> {f["price_estimate"]:.2f} €, nearest list price {f["price_point_estimate"]:.2f} € '
              f'(80% range {f["price_low"]:.2f} to {f["price_high"]:.2f} €; {html.escape(f["price_estimate_kind"])}; '
              f'list {r["PRICE_BASE_W_VAT"]:.2f} €). Main effects: {html.escape(f["price_reasons"])}.</p>')
    return (f'<section><h3>{html.escape(q)}</h3><div class="row">{card(d, q, "<em>query</em>")}'
            f'<div class="arrow">→</div>{"".join(cards)}</div>{fc}</section>')


def cmd_report(args: argparse.Namespace) -> None:
    d = load()
    if d["ex"] is None:
        raise SystemExit("Run the explanations first: python src/phase5_explain.py build")
    rng = np.random.default_rng(0)
    it = d["items"]
    picks = [BRIEF_QUERY]
    for _cat, grp in it.groupby("CAT_DES_EN"):
        with_photo = sorted(grp.index[grp["img_trusted"].fillna(False).astype(bool)])
        picks += list(rng.choice(with_photo, PER_CATEGORY, replace=False))
    no_photo = sorted(it.index[~it["img_trusted"].fillna(False).astype(bool)])
    picks += list(rng.choice(no_photo, 1, replace=False))
    bc = brief_check(d)
    rows = "".join(f"<tr><td>{r['brief answer']}</td><td>{r['engine rank']}</td><td>{r['out of']}</td></tr>"
                   for _, r in bc.iterrows())
    page = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parfois similar products</title>
<style>
:root{{--bg:#f7f7f8;--fg:#1d2230;--muted:#5d6475;--card:#fff;--rule:#dde0e6;--accent:#6b3b5f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15181e;--fg:#e6e8ee;--muted:#a3a9b6;--card:#1d212a;--rule:#2f3542;--accent:#d8a6c9}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1200px;margin:auto}} h1{{margin:0 0 4px}} .lede{{color:var(--muted);max-width:70ch}}
section{{border-top:1px solid var(--rule);padding:14px 0}} h3{{margin:0 0 8px;font-family:ui-monospace,Consolas,monospace}}
.row{{display:flex;gap:10px;overflow-x:auto;align-items:flex-start;padding-bottom:6px}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:8px;min-width:170px;max-width:190px;
display:flex;flex-direction:column;gap:3px}} .card img,.noimg{{width:150px;height:150px;object-fit:contain;align-self:center;background:#fff}}
.noimg{{display:grid;place-items:center;color:#888;background:var(--bg)}} .card p{{margin:2px 0;font-size:12px}}
.diff{{color:var(--muted)}} .arrow{{align-self:center;font-size:24px;color:var(--muted)}} em{{color:var(--accent)}}
.bar{{display:flex;height:8px;border-radius:4px;overflow:hidden;background:var(--rule)}} .bar i{{display:block;height:100%}}
.s0{{background:#3b6ea5}}.s1{{background:#c8873a}}.s2{{background:#6b9e5a}}.s3{{background:#8b5fa8}}.s4{{background:#c45c6a}}.s5{{background:#7a8291}}
.fc{{color:var(--muted);font-size:13px}} table{{border-collapse:collapse}} td,th{{border:1px solid var(--rule);padding:4px 10px}}
</style><main>
<h1>Parfois: similar products, explained</h1>
<p class="lede">For each product: its 5 most similar products in the same category (never another colour of the same
model), the share of the score that comes from each kind of information (bar), what they share and how they differ,
and the sales and price estimates of Phase 5b (never fitted on the product itself) with an 80% range and main effects. Generated {datetime.now(timezone.utc):%Y-%m-%d}
by <code>src/phase6_deliver.py report</code> from the batch files.</p>
<h2>The brief's example</h2><p>The brief gives four similar products for {BRIEF_QUERY}. Their rank in our engine
(out of all candidates of the category):</p><table><tr><th>brief answer</th><th>engine rank</th><th>out of</th></tr>{rows}</table>
{''.join(query_block(d, q) for q in picks)}
</main>"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.html").write_text(page, encoding="utf-8")
    bc.to_csv(REPORT_DIR / "brief_check.csv", index=False)
    print(bc.to_string(index=False))
    print(f"-> {REPORT_DIR / 'report.html'} ({len(picks)} products)")


BRIEF_COLS = {"GFS_DES_EN": "sub-family", "THEME": "theme", "FINISHING": "finish", "CLR_DES": "colour",
              "PRICE_BASE_W_VAT": "price €", "SEA_COD": "season"}


def cmd_brief(args: argparse.Namespace) -> None:
    """The brief's example next to the engine's answer: photos side by side and the attributes that differ."""
    from PIL import Image, ImageDraw
    d = load()
    it, ex = d["items"], d["ex"]
    if ex is None:
        raise SystemExit("Run the explanations first: python src/phase5_explain.py build")
    engine_top = ex[ex["query"] == BRIEF_QUERY].sort_values("rank")["candidate"].tolist()
    bc = brief_check(d).set_index("brief answer")
    rows = [("query", BRIEF_QUERY)] + [("brief", k) for k in BRIEF_ANSWER] + [("engine", k) for k in engine_top]
    size, pad = 220, 30
    strip = Image.new("RGB", (size * len(rows), size + pad), "white")
    draw = ImageDraw.Draw(strip)
    for n, (who, k) in enumerate(rows):
        file = it.at[k, "img_file"]
        if isinstance(file, str):
            im = Image.open(IMG_DIR / file).convert("RGB")
            im.thumbnail((size - 10, size - 10))
            strip.paste(im, (n * size + (size - im.width) // 2, (size - im.height) // 2))
        else:
            draw.text((n * size + 60, size // 2), "no trusted photo", fill="grey")
        draw.text((n * size + 8, size + 6), f"{who}: {k}", fill="black")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    strip.save(REPORT_DIR / "brief_example.jpg", quality=85)
    t = it.loc[[k for _, k in rows], list(BRIEF_COLS)].rename(columns=BRIEF_COLS)
    t.insert(0, "in", [w for w, _ in rows])
    t["model"] = it.loc[t.index, "PROD_REF"].astype(str)
    t["engine rank"] = ["–"] + [str(bc.at[k, "engine rank"]) for k in BRIEF_ANSWER] + [str(r) for r in
                                                                                       range(1, len(engine_top) + 1)]
    q = it.loc[BRIEF_QUERY]

    def count(w, c):
        return int((it.loc[[k for who, k in rows if who == w], c] == q[c]).sum())
    md = ["# The brief's example: 167718_BU", "",
          f"Generated by `src/phase6_deliver.py brief` ({datetime.now(timezone.utc):%Y-%m-%d}) from the Phase 3 "
          "neighbours and Phase 5 explanations. Photos: `brief_example.jpg` (query, the brief's four answers, the "
          "engine's top 5).", "",
          "![The brief's example and the engine's answer](brief_example.jpg)", "",
          "| | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
    md += [f"| {k} | " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |"
           for k, r in t.iterrows()]
    md += ["", "What matches the query (out of 4 brief answers / 5 engine answers):", "",
           "| | brief | engine |", "|---|---|---|"]
    md += [f"| same {lab} | {count('brief', c)} / 4 | {count('engine', c)} / {len(engine_top)} |"
           for c, lab in [("GFS_DES_EN", "sub-family"), ("THEME", "theme"), ("FINISHING", "finish")]]
    n_models = {w: len({it.at[k, 'PROD_REF'] for who, k in rows if who == w}) for w in ("brief", "engine")}
    md += [f"| different models | {n_models['brief']} | {n_models['engine']} |", "",
           "- The brief lists `117781_LB`, which is not in the data; `177781_LB` is (one digit differs), and its "
           "photo is the one on the slide, so it is used here.",
           f"- The engine ranks the brief's answers {' / '.join(str(bc.at[k, 'engine rank']) for k in BRIEF_ANSWER)} "
           f"of {int(bc['out of'].iloc[0]):,} candidates of the category.",
           "- Both lists are beaded earrings (see the photos). The engine's are dangle earrings with the query's "
           "golden finish and price; the brief's are mostly another theme (DEGRADÊ), finish *Other*, and one is a "
           "stud. The difference is one of criterion, not an error of either list.",
           "- Both lists repeat a model in several colours (the brief 169607 twice, the engine 210347 three times); "
           "whether colours of one model should count as one suggestion is open (a question for the professor).",
           "- One example cannot evaluate an engine: it is a sanity check of the format. The evaluation is the "
           "team's sorted answers on 99 queries (Phase 4).", ""]
    (REPORT_DIR / "brief_example.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("query")
    q.add_argument("key")
    q.add_argument("--k", type=int, default=5)
    sub.add_parser("report")
    sub.add_parser("brief")
    args = ap.parse_args()
    {"query": cmd_query, "report": cmd_report, "brief": cmd_brief}[args.cmd](args)


if __name__ == "__main__":
    main()
