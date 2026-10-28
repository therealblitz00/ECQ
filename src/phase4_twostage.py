"""Phase 4: two-stage retrieval and the other proposals of the external audit, on the usual signals.

    python src/phase4_twostage.py   # -> outputs/phase4/twostage_check.md + twostage_check.csv, ~5 min

Variants of the current engine (`src/phase3_weights.json`), nothing else changed:

- cap: at most 1 colour per model in the top k (re-ranked inside the top 200);
- soft: the categorical weights of colour, finishing and theme lowered to 0.5 (from 2.0 / 1.5 / 1.5);
- two-stage N: the N nearest photos of the same sub-family (other models) come first, ordered by a re-rank score
  (the engine without the image block, or the full engine score); everything else follows in the usual order. A query
  without a trusted photo keeps the usual engine.

Signals, as in phase4_compare: the team's preferences with the current weights (share of the sorted pairs ordered right,
paired bootstrap over queries against the baseline), the guarded metrics of phase4_eval on the test half of the models,
the number of distinct models in the top 5 and the brief's example. Read-only: the engine and its outputs are unchanged.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
import phase3_engine as engine  # noqa: E402
import phase4_compare as cmp  # noqa: E402
import phase4_eval as ev  # noqa: E402
import phase4_feedback as fb  # noqa: E402
from config import KEY, PROCESSED_DIR, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase4"
SOFT = {"CLR_DES": 0.5, "FINISHING": 0.5, "THEME": 0.5}
CAP_WINDOW = 200     # the cap re-ranks inside the top 200 of the usual order
VARIANTS = {
    "baseline (current engine)": {},
    "cap 1 colour per model": {"cap": 1},
    "soft colour/finish/theme + cap": {"soft": True, "cap": 1},
    "two-stage 50, re-rank without the photo": {"two_stage": 50},
    "two-stage 50, re-rank with the full score": {"two_stage": 50, "rerank": "full"},
    "two-stage 20, re-rank without the photo": {"two_stage": 20},
    "two-stage 50 + soft + cap (audit's best)": {"two_stage": 50, "soft": True, "cap": 1},
}


def soft_categorical(blocks: dict, scale: dict[str, float]) -> np.ndarray:
    """The categorical block with the named attributes re-weighted (rows renormalised)."""
    attrs = json.loads((PROCESSED_DIR / "feature_meta.json").read_text(encoding="utf-8"))["blocks"]["categorical"][
        "attributes"]
    m = blocks["categorical"].copy()
    for name, w in scale.items():
        m[:, attrs[name]["start"]:attrs[name]["stop"]] *= np.sqrt(w / attrs[name]["weight"])
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


def ranked(items: pd.DataFrame, blocks: dict, weights: dict, queries: list[str], k: int = 10, two_stage: int = 0,
           rerank: str = "no_image", soft: bool = False, cap: int = 0) -> pd.DataFrame:
    """Top-k of every query under one variant: one row per (query, rank, candidate)."""
    bl = dict(blocks)
    if soft:
        bl["categorical"] = soft_categorical(blocks, SOFT)
    has = engine.has_block(bl)
    w_rr = {b: w for b, w in weights.items() if b != "image"} if rerank == "no_image" else weights
    keys = items[KEY].to_numpy()
    gfs = (items["GFA_DES_EN"].astype(str) + " › " + items["GFS_DES_EN"].astype(str)).to_numpy()
    model = items["PROD_REF"].astype(str).to_numpy()
    photo = has["image"] > 0
    wanted = items[KEY].isin(queries).to_numpy()
    frames = []
    for pool in engine.groups(items, "category"):
        qs = pool[wanted[pool]]
        for start in range(0, len(qs), engine.CHUNK):
            q = qs[start:start + engine.CHUNK]
            s = np.round(engine.score_matrix(bl, has, weights, q, pool).astype(np.float64), engine.DECIMALS)
            excl = engine.excluded(items, q, pool, include_siblings=False)
            if two_stage:
                ok = (gfs[q][:, None] == gfs[pool][None, :]) & ~excl & photo[pool][None, :] & photo[q][:, None]
                im = np.where(ok, bl["image"][q] @ bl["image"][pool].T, -np.inf)
                stage = np.zeros_like(ok)
                np.put_along_axis(stage, np.argsort(-im, axis=1, kind="stable")[:, :two_stage], True, axis=1)
                rr = np.round(engine.score_matrix(bl, has, w_rr, q, pool).astype(np.float64), engine.DECIMALS)
                s = np.where(stage & ok, 10 + rr, s)   # scores are at most 1: stage 1 always comes first
            s[excl] = -np.inf
            order = np.argsort(-s, axis=1, kind="stable")[:, :max(k, CAP_WINDOW)]
            for i, qi in enumerate(q):
                cand = pool[order[i][np.isfinite(s[i, order[i]])]]
                if cap:
                    seen: dict[str, int] = {}
                    kept, rest = [], []
                    for c in cand[:CAP_WINDOW]:
                        (kept if seen.get(model[c], 0) < cap else rest).append(c)
                        seen[model[c]] = seen.get(model[c], 0) + 1
                    cand = np.concatenate([np.asarray(kept + rest, dtype=int), cand[CAP_WINDOW:]])
                cand = cand[:k]
                frames.append(pd.DataFrame({"query": keys[qi], "rank": np.arange(1, len(cand) + 1),
                                            "candidate": keys[cand]}))
    return pd.concat(frames, ignore_index=True)


def team_table(nb: pd.DataFrame, recs: list[dict]) -> pd.DataFrame:
    """Right / total sorted pairs per query: each similar item above the similar items below it and every "not"."""
    rank = nb.set_index(["query", "candidate"])["rank"].to_dict()
    out: dict[str, list[int]] = {}
    for r in recs:
        o = out.setdefault(r["query"], [0, 0])
        for i, a in enumerate(r["similar"]):
            for b in r["similar"][i + 1:] + r["not"]:
                o[0] += int(rank.get((r["query"], a), 10**6) < rank.get((r["query"], b), 10**6))
                o[1] += 1
    return pd.DataFrame(out, index=["right", "pairs"]).T


def judged_coverage(nb: pd.DataFrame, recs: list[dict]) -> float:
    """Share of the top 5 the team judged (its pools came from the current engine)."""
    judged = {(r["query"], c) for r in recs for c in r["similar"] + r["not"]}
    t = nb[nb["rank"] <= 5]
    return float(np.mean([(q, c) in judged for q, c in zip(t["query"], t["candidate"])]))


def run() -> pd.DataFrame:
    items = phase1_clean.load_items()
    blocks, weights = engine.load_blocks(items), engine.load_weights()
    recs = fb.read_feedback()
    team_q = sorted({r["query"] for r in recs})
    test_q = items.loc[~ev.split_models(items), KEY].tolist()
    gid, pools = ev.known_groups(items), ev.visual_pools(items, blocks["image"])
    model = items.set_index(KEY)["PROD_REF"].astype(str)
    rows, tables = {}, {}
    for name, kw in VARIANTS.items():
        nb_team = ranked(items, blocks, weights, team_q, k=len(items), **kw)
        tables[name] = team_table(nb_team, recs)
        nb_test = ranked(items, blocks, weights, test_q, **kw)
        m = ev.metrics(items, nb_test, gid, pools)
        top5 = nb_test[nb_test["rank"] <= 5]
        brief = ranked(items, blocks, weights, [cmp.BRIEF_QUERY], k=len(items), **kw).set_index("candidate")["rank"]
        rows[name] = {"team pairs right": cmp.accuracy(tables[name]),
                      "top 5 judged by the team": judged_coverage(nb_team, recs),
                      **{g: m[g] for g in ev.GUARDED},
                      "distinct models in the top 5": float(top5.assign(m=model.loc[top5["candidate"]].to_numpy())
                                                            .groupby("query")["m"].nunique().mean()),
                      "brief ranks": " / ".join(str(int(brief.get(c, -1))) for c in cmp.BRIEF_ANSWER)}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[name].items()}, flush=True)
    df = pd.DataFrame(rows).T
    base = next(iter(VARIANTS))
    ci = {n: cmp.boot_diff(tables[base], tables[n]) for n in VARIANTS if n != base}
    write_report(df, ci, base)
    df.assign(**{"team diff low": [ci.get(n, (np.nan,) * 3)[0] for n in df.index],
                 "team diff high": [ci.get(n, (np.nan,) * 3)[1] for n in df.index]}).to_csv(
        REPORT_DIR / "twostage_check.csv")
    return df


def write_report(df: pd.DataFrame, ci: dict, base: str) -> None:
    def fmt(c, v):
        if isinstance(v, str):
            return v
        return f"{v:.3f}" if c == "sales_corr" else (f"{v:.2f}" if c.startswith("distinct") else f"{100 * v:.1f}%")

    cols = ["team pairs right", "hit@4", "same_subfamily", "same_colour", "same_finishing", "same_material",
            "visual@4", "sales_corr", "distinct models in the top 5", "top 5 judged by the team", "brief ranks"]
    lines = ["| Variant | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for n, r in df.iterrows():
        team = fmt(cols[0], r[cols[0]])
        if n in ci:
            lo, hi, p = ci[n]
            team += f" ({100 * lo:+.1f} to {100 * hi:+.1f})"
        lines.append(f"| {n} | {team} | " + " | ".join(fmt(c, r[c]) for c in cols[1:]) + " |")
    guard_fail = {n: [g for g in ev.GUARDED if df.at[n, g] < df.at[base, g] - ev.TOLERANCE] for n in df.index if n != base}
    fails = "\n".join(f"- {n}: {', '.join(g) or 'none'}" for n, g in guard_fail.items())
    text = f"""# Two-stage retrieval and the audit's other proposals

Generated by `src/phase4_twostage.py` ({datetime.now(timezone.utc):%Y-%m-%d}). Team: share of the sorted pairs ordered
right with the current weights, in brackets the 95% interval of the difference to the baseline (paired bootstrap over
queries, points). Guards: phase4_eval on the test half of the models. Brief ranks: 169597_FU / 169607_IG / 169607_PE /
177781_LB of 4,638.

{chr(10).join(lines)}

Guards down by more than {100 * ev.TOLERANCE:.0f} point:

{fails}

- The team's pools were built with the current engine, so the variants put unjudged items in their top 5 (the coverage
  column): the team signal sees less of them.
- `177781_LB` is a stud and the query a dangle earring: a first stage restricted to the sub-family cannot bring it
  forward.
"""
    (REPORT_DIR / "twostage_check.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
