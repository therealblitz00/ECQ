"""Phase 4: interactive human feedback. Reviewers sort candidates in the browser; the block weights learn from it.

    python src/phase4_feedback.py pool              # one person, once: fixed candidate pools -> outputs/phase4/feedback_pool.json
    python src/phase4_feedback.py app --id 4        # each member: review app on http://localhost:8766
    python src/phase4_feedback.py fit               # learned weights + held-out check -> outputs/phase4/feedback_report.md
    python src/phase4_feedback.py fit --apply       # also write them to src/phase3_weights.json (then phase3_engine.py build)

How it learns: for each query the reviewer drags the candidates into "Similar" (best first) or "Not similar". That
gives pairwise preferences (every similar item beats the ones below it and every "not" item). The engine score is
`score = sum_b w_b * sim_b / sum_b w_b` over the blocks both items have, so the six weights are fitted with a
Bradley-Terry (logistic) model on those preferences, pulled towards the current weights (a Gaussian prior on the
log-weights), so a few odd clicks cannot swing them. The app refits after every query and shows the change.

Why the candidates never depend on the learned weights (no feedback loop):

- The pool of each query is built once (`pool`) and committed: the top 5 at image weights 0, 0.3 and 0.6, plus
  RANDOM_PER_QUERY random items of the same category, so every weight setting is judged on the same items.
- The candidates are shown in a fixed shuffled order, without scores or source.
- HOLDOUT of the queries are never fitted: they only measure the learned weights against the starting ones.

Each member writes only `outputs/phase4/feedback_<n>.jsonl` (append-only; the latest answer per query counts).
A few queries are shared by every member, to measure how much reviewers agree.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
import phase3_engine as engine  # noqa: E402
import phase4_eval as ev  # noqa: E402
from config import IMG_DIR, KEY, ROOT, TEAM  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase4"
POOL_FILE = REPORT_DIR / "feedback_pool.json"
FEEDBACK_PATTERN = "feedback_*.jsonl"
SEED = 0
QUERIES_PER_CAT = (16, 4)   # with / without a trusted photo
POOL_IMAGE = (0.0, 0.3, 0.6)
POOL_TOP = 5
RANDOM_PER_QUERY = 2        # random same-category items: lets the fit see weights outside the three settings
SHARED = 10                 # queries every member sorts (reviewer agreement)
HOLDOUT = 0.2               # share of queries never fitted
PRIOR_SD = 0.5              # prior on log-weights: a weight moves by a factor ~1.6 only with clear evidence
BETA0 = 20.0                # start of the score scale: pool scores differ by a few hundredths
STEPS, LR = 1500, 0.05
MIN_HOLDOUT_PAIRS = 30


# --------------------------------------------------------------------------- pool (built once)
def choose_queries(items: pd.DataFrame, seed: int = SEED) -> list[str]:
    """Per category: random queries with a trusted photo and some without (fixed seed), sorted by category."""
    rng = np.random.default_rng(seed)
    picked = []
    for _, grp in items.groupby("CAT_DES_EN", sort=True):
        photo = grp["img_trusted"].fillna(False).astype(bool)
        for mask, n in zip((photo, ~photo), QUERIES_PER_CAT):
            keys = np.sort(grp.loc[mask, KEY].to_numpy())
            picked += sorted(rng.choice(keys, size=min(n, len(keys)), replace=False))
    return picked


def assign(queries: list[str], members: list[int], seed: int = SEED) -> tuple[dict[str, list[int]], set[str]]:
    """Members of each query (SHARED queries go to everyone, the rest round-robin) and the held-out queries."""
    rng = np.random.default_rng(seed)
    order = [queries[i] for i in rng.permutation(len(queries))]
    who = {q: list(members) for q in order[:SHARED]}
    who.update({q: [members[i % len(members)]] for i, q in enumerate(order[SHARED:])})
    holdout = set(rng.choice(sorted(queries), size=round(HOLDOUT * len(queries)), replace=False).tolist())
    return who, holdout


def build_pool(items: pd.DataFrame, blocks: dict, weights: dict, queries: list[str], members: list[int],
               seed: int = SEED) -> dict:
    """Fixed candidates of every query, in shuffled display order, with their per-block similarities."""
    configs = {f"image {w:g}": ev.normalised(ev.with_image(weights, w)) for w in POOL_IMAGE}
    pooled = pd.concat([engine.neighbours(items, blocks, w, k=POOL_TOP, queries=queries) for w in configs.values()])
    pooled = pooled.groupby("query")["candidate"].agg(lambda s: sorted(set(s)))
    rng = np.random.default_rng(seed)
    pos = pd.Series(np.arange(len(items)), index=items[KEY])
    keys = items[KEY].to_numpy()
    cats = items["CAT_DES_EN"].to_numpy()
    who, holdout = assign(queries, members, seed)
    has = engine.has_block(blocks)
    out = []
    for q in queries:
        qi = int(pos[q])
        same = np.flatnonzero(cats == cats[qi])
        ok = same[~engine.excluded(items, np.array([qi]), same, include_siblings=False)[0]]
        rest = np.setdiff1d(keys[ok], pooled.get(q, []))
        extra = sorted(rng.choice(rest, size=min(RANDOM_PER_QUERY, len(rest)), replace=False).tolist())
        cands = list(pooled.get(q, [])) + extra
        cands = [cands[i] for i in rng.permutation(len(cands))]
        ci = pos.loc[cands].to_numpy()
        sims = engine.pair_breakdown(blocks, has, {b: 1.0 for b in weights}, np.full(len(ci), qi), ci)
        out.append({"query": q, "holdout": q in holdout, "members": who[q], "candidates": [
            {"key": c, "random": c in extra,
             "sim": {b: (None if np.isnan(sims[f"sim_{b}"][i]) else round(float(sims[f"sim_{b}"][i]), 6))
                     for b in weights}} for i, c in enumerate(cands)]})
    return {"blocks": list(weights), "weights": ev.normalised(weights), "configs": configs, "top": POOL_TOP,
            "random_per_query": RANDOM_PER_QUERY, "seed": seed, "queries": out,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def load_pool(path: Path = POOL_FILE) -> dict:
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} not found. Run once: python src/phase4_feedback.py pool")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- feedback files
def feedback_path(member: int) -> Path:
    return REPORT_DIR / f"feedback_{member}.jsonl"


def read_feedback(paths: list[Path] | None = None) -> list[dict]:
    """Latest answer of each member for each query (files are append-only, so an answer can be redone)."""
    paths = sorted(REPORT_DIR.glob(FEEDBACK_PATTERN)) if paths is None else paths
    latest: dict[tuple, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r["member"], r["query"])] = r
    return [latest[k] for k in sorted(latest)]


def check_answer(pool_q: dict, similar: list[str], not_similar: list[str]) -> None:
    """Every candidate placed exactly once, nothing else."""
    placed = list(similar) + list(not_similar)
    expected = {c["key"] for c in pool_q["candidates"]}
    if len(placed) != len(set(placed)) or set(placed) != expected:
        raise ValueError("place every candidate exactly once (Similar or Not similar)")


# --------------------------------------------------------------------------- learning
def sim_matrix(pool: dict) -> tuple[dict[tuple[str, str], int], np.ndarray, np.ndarray]:
    """Row index of every (query, candidate), the similarities (0 where missing) and the has-block mask."""
    blocks = pool["blocks"]
    index, rows = {}, []
    for q in pool["queries"]:
        for c in q["candidates"]:
            index[(q["query"], c["key"])] = len(rows)
            rows.append([np.nan if c["sim"][b] is None else c["sim"][b] for b in blocks])
    s = np.asarray(rows, dtype=float).reshape(-1, len(blocks))
    h = np.isfinite(s).astype(float)
    return index, np.nan_to_num(s), h


def preference_pairs(records: list[dict], index: dict) -> tuple[np.ndarray, np.ndarray]:
    """(winner rows, loser rows): each similar item beats the similar items below it and every "not" item."""
    win, lose = [], []
    for r in records:
        sim = [index[(r["query"], c)] for c in r["similar"]]
        nots = [index[(r["query"], c)] for c in r["not"]]
        for i, a in enumerate(sim):
            for b in sim[i + 1:] + nots:
                win.append(a)
                lose.append(b)
    return np.asarray(win, dtype=int), np.asarray(lose, dtype=int)


def scores(w: np.ndarray, s: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The engine score: weighted mean of the similarities over the blocks both items have."""
    den = h @ w
    return np.divide((h * s) @ w, den, out=np.zeros(len(s)), where=den > 0)


def _softmax(t: np.ndarray) -> np.ndarray:
    e = np.exp(t - t.max())
    return e / e.sum()


def loss_and_grad(params: np.ndarray, s: np.ndarray, h: np.ndarray, win: np.ndarray, lose: np.ndarray,
                  theta0: np.ndarray) -> tuple[float, np.ndarray]:
    """Bradley-Terry negative log-likelihood of the preferences + Gaussian prior; params = (log-weights, log-scale)."""
    theta, gamma = params[:-1], params[-1]
    w, beta = _softmax(theta), np.exp(gamma)
    den = h @ w
    sc = np.divide((h * s) @ w, den, out=np.zeros(len(s)), where=den > 0)
    d = sc[win] - sc[lose]
    loss = np.logaddexp(0.0, -beta * d).sum()
    loss += ((theta - theta0) ** 2).sum() / (2 * PRIOR_SD ** 2) + (gamma - np.log(BETA0)) ** 2 / 2
    g_d = -beta / (1.0 + np.exp(np.clip(beta * d, -50, 50)))            # dloss / d(score difference)
    ds_dw = np.divide(h * s - sc[:, None] * h, den[:, None], out=np.zeros_like(s), where=den[:, None] > 0)
    g_w = (g_d[:, None] * (ds_dw[win] - ds_dw[lose])).sum(axis=0)
    g_theta = w * (g_w - w @ g_w) + (theta - theta0) / PRIOR_SD ** 2
    g_gamma = (g_d * d).sum() + (gamma - np.log(BETA0))
    return float(loss), np.append(g_theta, g_gamma)


def fit_weights(s: np.ndarray, h: np.ndarray, win: np.ndarray, lose: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """Maximum a posteriori weights (Adam, fixed steps: deterministic). Returns weights summing to 1."""
    theta0 = np.log(np.maximum(w0, 1e-3) / np.maximum(w0, 1e-3).sum())
    if len(win) == 0:
        return _softmax(theta0)
    p = np.append(theta0, np.log(BETA0))
    m, v = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, STEPS + 1):
        _, g = loss_and_grad(p, s, h, win, lose, theta0)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return _softmax(p[:-1])


def holdout_metrics(w: np.ndarray, s: np.ndarray, h: np.ndarray, records: list[dict], index: dict) -> dict:
    """Share of the reviewers' preferences the weights get right, and the share of the top 3 they called similar."""
    win, lose = preference_pairs(records, index)
    sc = scores(w, s, h)
    top3 = []
    for r in records:
        cands = r["similar"] + r["not"]
        order = sorted(cands, key=lambda c: (-round(sc[index[(r["query"], c)]], 6), c))
        top3 += [c in r["similar"] for c in order[:3]]
    # None, not NaN, when there is nothing to measure: the page reads this as JSON
    return {"pairs": int(len(win)), "pair_accuracy": float((sc[win] > sc[lose]).mean()) if len(win) else None,
            "similar@3": float(np.mean(top3)) if top3 else None}


def learn(pool: dict, records: list[dict]) -> dict:
    """Fit on the non-held-out answers; measure start and learned weights on the held-out ones."""
    index, s, h = sim_matrix(pool)
    blocks = pool["blocks"]
    w0 = np.array([pool["weights"][b] for b in blocks])
    hold = {q["query"] for q in pool["queries"] if q["holdout"]}
    train = [r for r in records if r["query"] not in hold]
    test = [r for r in records if r["query"] in hold]
    win, lose = preference_pairs(train, index)
    w = fit_weights(s, h, win, lose, w0)
    return {"blocks": blocks, "start": dict(zip(blocks, np.round(w0, 4).tolist())),
            "learned": dict(zip(blocks, np.round(w, 4).tolist())), "pairs": int(len(win)),
            "queries": len({r["query"] for r in train}),
            "holdout": {"start": holdout_metrics(w0, s, h, test, index),
                        "learned": holdout_metrics(w, s, h, test, index)}}


def agreement(records: list[dict]) -> dict:
    """On queries sorted by several members: share of candidate pairs that two members order the same way."""
    by_q: dict[str, list[dict]] = {}
    for r in records:
        by_q.setdefault(r["query"], []).append(r)
    same = total = 0
    for rs in by_q.values():
        ranks = [{c: i for i, c in enumerate(r["similar"])} | {c: len(r["similar"]) for c in r["not"]} for r in rs]
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                cands = sorted(ranks[i])
                for x in range(len(cands)):
                    for y in range(x + 1, len(cands)):
                        a = np.sign(ranks[i][cands[x]] - ranks[i][cands[y]])
                        b = np.sign(ranks[j][cands[x]] - ranks[j][cands[y]])
                        if a != 0 and b != 0:
                            same, total = same + (a == b), total + 1
    return {"pairs": total, "agreement": same / total if total else None}


# --------------------------------------------------------------------------- report and apply
def write_report(res: dict, records: list[dict], agree: dict, applied: str) -> None:
    hs, hl = res["holdout"]["start"], res["holdout"]["learned"]
    wrows = "\n".join(f"| `{b}` | {res['start'][b]:.3f} | {res['learned'][b]:.3f} |" for b in res["blocks"])
    pct = lambda v: "–" if pd.isna(v) else f"{v:.0%}"  # noqa: E731
    members = sorted({r["member"] for r in records})
    text = f"""# Phase 4: interactive feedback report

Generated by `src/phase4_feedback.py fit` ({datetime.now(timezone.utc):%Y-%m-%d}) from
{len(records)} sorted queries ({', '.join(f'{m} {TEAM.get(m, "?")}' for m in members) or 'nobody yet'}).

The reviewers drag each query's candidates into *Similar* (best first) or *Not similar*. Every similar item beats the
items below it and every "not" item; the six block weights are fitted to those preferences with a Bradley-Terry
model, pulled towards the starting weights (prior sd {PRIOR_SD} on the log-weights). Candidates come from a fixed pool
(top {POOL_TOP} at image weights {', '.join(f'{w:g}' for w in POOL_IMAGE)} + {RANDOM_PER_QUERY} random), shown shuffled and
blind, so the learned weights never change what a reviewer sees.

## Weights

Fitted on {res['queries']} queries ({res['pairs']:,} preferences).

| Block | Start | Learned |
|---|---|---|
{wrows}

## Held-out check

{round(HOLDOUT * 100)}% of the queries are never fitted. On them:

| Weights | Preferences | Ordered right | Top 3 judged similar |
|---|---|---|---|
| start | {hs['pairs']} | {pct(hs['pair_accuracy'])} | {pct(hs['similar@3'])} |
| learned | {hl['pairs']} | {pct(hl['pair_accuracy'])} | {pct(hl['similar@3'])} |

## Reviewer agreement

On the {SHARED} queries every member sorts: {agree['pairs']} candidate pairs ordered by two members,
{pct(agree['agreement'])} in the same order. This is the ceiling: the weights cannot agree with "the team" more than
its members agree with each other.

## Applied

{applied}
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "feedback_report.md").write_text(text, encoding="utf-8")


def apply_weights(res: dict) -> str:
    """Write the learned weights, only if they do at least as well as the start on enough held-out preferences."""
    hs, hl = res["holdout"]["start"], res["holdout"]["learned"]
    if hl["pairs"] < MIN_HOLDOUT_PAIRS:
        return f"Not applied: only {hl['pairs']} held-out preferences (at least {MIN_HOLDOUT_PAIRS} needed)."
    if hl["pair_accuracy"] < hs["pair_accuracy"]:
        return "Not applied: the learned weights order fewer held-out preferences right than the start."
    meta = json.loads(engine.WEIGHTS_FILE.read_text(encoding="utf-8"))
    meta["weights"] = res["learned"]
    meta["note"] = ("Block weights of the Phase 3 score (score = sum of w_b * sim_b over the blocks both items have, "
                    "divided by the sum of those w_b). Learned from the team's sorting by src/phase4_feedback.py "
                    "(outputs/phase4/feedback_report.md).")
    meta["tuned"] = f"phase4_feedback.py fit --apply, {datetime.now(timezone.utc):%Y-%m-%d}"
    engine.WEIGHTS_FILE.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return ("Written to `src/phase3_weights.json`. Next: `python src/phase3_engine.py build`, then "
            "`python src/phase4_eval.py run` to check that no catalogue, visual or sales metric dropped.")


# --------------------------------------------------------------------------- review app
def card(it: pd.Series) -> dict:
    trusted = bool(it["img_trusted"]) if pd.notna(it["img_trusted"]) else False
    val = lambda c: "" if pd.isna(it[c]) else str(it[c])  # noqa: E731
    return {"key": it.name, "img": it["img_file"] if trusted and val("img_file") else "",
            "desc": val("PROD_DES_BASE"), "family": val("GFA_DES_EN"), "subfamily": val("GFS_DES_EN"),
            "colour": val("CLR_DES"), "finishing": val("FINISHING"), "material": val("MATERIAL"),
            "composition": val("COMPOSITION"), "price": f"{it['PRICE_BASE_W_VAT']:.2f}"}


class Session:
    """One member's queries, answers and the live learning state."""

    def __init__(self, member: int):
        self.member, self.name = member, TEAM[member]
        self.pool = load_pool()
        self.path = feedback_path(member)
        self.lock = threading.Lock()
        mine = [q for q in self.pool["queries"] if member in q["members"]]
        it = phase1_clean.load_items().set_index(KEY)
        self.queries = [{"query": q["query"], "item": card(it.loc[q["query"]]),
                         "candidates": [card(it.loc[c["key"]]) for c in q["candidates"]]} for q in mine]
        self.by_query = {q["query"]: q for q in self.pool["queries"]}
        self.state = learn(self.pool, read_feedback())

    def answers(self) -> dict:
        return {r["query"]: r for r in read_feedback([self.path])}

    def save(self, query: str, similar: list[str], not_similar: list[str], seconds: float) -> dict:
        q = self.by_query.get(query)
        if q is None or self.member not in q["members"]:
            raise ValueError(f"{query} is not one of your queries")
        check_answer(q, similar, not_similar)
        rec = {"member": self.member, "reviewer": self.name, "query": query, "similar": similar, "not": not_similar,
               "seconds": round(float(seconds), 1), "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.state = learn(self.pool, read_feedback())
        return self.state


def make_handler(sess: Session):
    from review_app import git_submit

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False, default=float).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json({"member": sess.member, "name": sess.name, "file": sess.path.name,
                            "queries": sess.queries, "answers": sess.answers(), "learning": sess.state})
            elif path.startswith("/img/"):
                f = IMG_DIR / Path(unquote(path[5:])).name  # file name only: no directory traversal
                if not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                self._send(200, f.read_bytes(), "image/png" if f.suffix.lower() == ".png" else "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if path == "/api/save":
                    state = sess.save(body["query"], body["similar"], body["not"], body.get("seconds", 0))
                    self._json({"ok": True, "learning": state})
                elif path == "/api/submit":
                    ok, log = git_submit(sess.path, f"Phase 4: feedback {sess.member} ({sess.name})")
                    self._json({"ok": ok, "log": log})
                else:
                    self._json({"ok": False, "error": "unknown endpoint"}, 404)
            except Exception as e:  # report to the page instead of crashing the server
                self._json({"ok": False, "error": str(e)}, 400)

    return Handler


# --------------------------------------------------------------------------- CLI
def cmd_pool(args: argparse.Namespace) -> None:
    if POOL_FILE.exists() and read_feedback() and not args.force:
        raise SystemExit("Feedback already exists for the current pool: not rebuilt (use --force to start over).")
    items, blocks = ev.load()
    pool = build_pool(items, blocks, engine.load_weights(), choose_queries(items), sorted(TEAM))
    POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    POOL_FILE.write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n = pd.Series([m for q in pool["queries"] for m in q["members"]]).value_counts().sort_index()
    print(f"{len(pool['queries'])} queries, {sum(len(q['candidates']) for q in pool['queries'])} candidates, "
          f"{sum(q['holdout'] for q in pool['queries'])} held out -> {POOL_FILE.relative_to(ROOT)}")
    print("queries per member:", ", ".join(f"{m} {TEAM[m]}: {c}" for m, c in n.items()))


def cmd_app(args: argparse.Namespace) -> None:
    if args.id not in TEAM:
        raise SystemExit(f"--id must be one of {list(TEAM)}")
    sess = Session(args.id)
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(sess))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port found.")
    url = f"http://localhost:{server.server_port}"
    print(f"Feedback app for {sess.name}: {len(sess.queries)} queries, running at {url}")
    print("Every answer is saved at once. Close this window when done.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_fit(args: argparse.Namespace) -> None:
    t = time.time()
    pool, records = load_pool(), read_feedback()
    res = learn(pool, records)
    applied = apply_weights(res) if args.apply else "Not applied (`--apply` writes the weights if the held-out check passes)."
    write_report(res, records, agreement(records), applied)
    print(json.dumps({k: res[k] for k in ("start", "learned", "pairs", "holdout")}, indent=1, default=float))
    print(applied, f"({time.time() - t:.1f} s) -> outputs/phase4/feedback_report.md")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pool", help="fixed candidate pools (one person, once)")
    s.add_argument("--force", action="store_true", help="rebuild even if feedback exists (old answers stop counting)")
    s.set_defaults(fn=cmd_pool)
    a = sub.add_parser("app", help="sort your queries in the browser")
    a.add_argument("--id", type=int, required=True, help="your team number: " +
                   ", ".join(f"{k}={v}" for k, v in TEAM.items()))
    a.add_argument("--port", type=int, default=8766)
    a.add_argument("--no-browser", action="store_true")
    a.set_defaults(fn=cmd_app)
    f = sub.add_parser("fit", help="learn the weights from everybody's feedback")
    f.add_argument("--apply", action="store_true", help="write them to src/phase3_weights.json if the check passes")
    f.set_defaults(fn=cmd_fit)
    args = p.parse_args()
    args.fn(args)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Similarity feedback</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--accent:#1f5f8b;--ok:#2e7d32;--bad:#c62828;--drop:#eef4f8}
*{box-sizing:border-box}body{margin:0;font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center}
h1{font-size:17px;margin:0}h2{font-size:14px;margin:0 0 6px}.small{font-size:12px;color:var(--muted)}
button,select{font:inherit;border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer;color:var(--ink)}
button:hover{border-color:var(--accent)}button:disabled{opacity:.45;cursor:default}.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;min-width:160px;flex:1}.bar>div{height:100%;background:var(--ok);width:0}
.help{background:#eef4f8;border-bottom:1px solid var(--line);padding:8px 16px;font-size:13px}
.layout{display:grid;grid-template-columns:240px 1fr 260px;gap:14px;padding:14px 16px;align-items:start}
@media (max-width:1100px){.layout{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.zone{min-height:150px;border:2px dashed var(--line);border-radius:10px;padding:8px;display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start;background:var(--card)}
.zone.over{background:var(--drop);border-color:var(--accent)}#similar{border-color:#9cc59e}#not{border-color:#e3a5a5}
.zones{display:grid;gap:12px}.zhead{display:flex;justify-content:space-between;align-items:baseline;margin:2px 2px 4px}
.card{width:168px;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:grab;display:flex;flex-direction:column;position:relative}
.card.dragging{opacity:.4}.imgbox{aspect-ratio:1;background:#fafafa;display:grid;place-items:center}
.imgbox img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in}.noimg{color:var(--muted);font-size:12px;padding:8px;text-align:center}
.cbody{padding:6px 7px;font-size:12px;display:flex;flex-direction:column;gap:2px}.cbody b{font-size:12.5px}
.btns{display:flex;gap:4px;padding:0 6px 6px}.btns button{flex:1;padding:3px 0}
.rank{position:absolute;top:4px;left:4px;background:var(--ok);color:#fff;border-radius:10px;padding:0 7px;font-weight:600;font-size:12px}
.query .card{width:100%;cursor:default}.query .cbody{font-size:13px}
.wrow{display:grid;grid-template-columns:86px 1fr 46px;gap:6px;align-items:center;font-size:12px;margin:4px 0}
.wbar{position:relative;height:12px;background:#f0f0ec;border-radius:3px}.wbar .s{position:absolute;top:0;bottom:0;border-right:2px solid #999}
.wbar .l{position:absolute;top:2px;bottom:2px;left:0;background:var(--accent);border-radius:2px}
.up{color:var(--ok)}.down{color:var(--bad)}
#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;place-items:center;z-index:10;cursor:zoom-out}#lightbox img{max-width:94vw;max-height:94vh;background:#fff}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1d1d1b;color:#fff;padding:8px 14px;border-radius:8px;display:none;z-index:11}
pre{white-space:pre-wrap;font-size:12px;max-height:300px;overflow:auto;background:#f0f0ec;padding:8px;border-radius:6px}
#dlg{position:fixed;inset:0;background:rgba(0,0,0,.4);display:none;place-items:center;z-index:12}#dlg>div{background:#fff;border-radius:10px;padding:16px;max-width:640px;width:92vw}
</style></head><body>
<header><h1 id="title">Similarity feedback</h1>
<button id="prev">‹ Previous</button><select id="jump"></select><button id="next">Next ›</button>
<div class="bar"><div id="bar"></div></div><span class="small" id="ptext"></span>
<button class="primary" id="submit">Submit my feedback</button></header>
<div class="help">Would a buyer show these next to the query? Drag each candidate (or use ✓ / ✗) into <b>Similar</b>, with the
<b>most similar first</b>, or into <b>Not similar</b>. Judge the product, not the photo; items without a trusted photo are judged from the text.
Click a photo to enlarge. When all are placed, <b>Save &amp; next</b>.</div>
<div class="layout">
 <div class="panel query"><h2>Query</h2><div id="qcard"></div></div>
 <div class="zones">
  <div><div class="zhead"><h2>To sort</h2><span class="small" id="left"></span></div><div class="zone" id="tray" data-zone="tray"></div></div>
  <div><div class="zhead"><h2>Similar: most similar first</h2><span class="small">drag to reorder</span></div><div class="zone" id="similar" data-zone="similar"></div></div>
  <div><div class="zhead"><h2>Not similar</h2><span class="small">order does not matter</span></div><div class="zone" id="not" data-zone="not"></div></div>
  <div style="display:flex;gap:8px;justify-content:flex-end"><button id="reset">Start this query again</button><button class="primary" id="save" disabled>Save &amp; next</button></div>
 </div>
 <div class="panel"><h2>What the engine is learning</h2><div class="small" style="margin-bottom:6px">Weight of each part of the score. Grey line = start, blue = learned from everybody's answers so far.</div>
  <div id="weights"></div><div class="small" id="lstats" style="margin-top:8px"></div></div>
</div>
<div id="lightbox"><img alt=""></div><div id="toast"></div>
<div id="dlg"><div><h3 style="margin-top:0">Submit my feedback</h3><div id="dlgMsg">Sending…</div><p><button id="closeDlg">Close</button></p></div></div>
<script>
const NAMES={image:"Photo",categorical:"Category",text:"Description",composition:"Materials",palette:"Colours",numeric:"Price & numbers"};
let D, cur=0, place={}, started=Date.now(), dragKey=null;
const $=s=>document.querySelector(s);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function toast(t){const e=$("#toast");e.textContent=t;e.style.display="block";clearTimeout(e._t);e._t=setTimeout(()=>e.style.display="none",1800)}
function cardHtml(c,zone,rank){
  const img=c.img?`<div class="imgbox"><img loading="lazy" src="/img/${encodeURIComponent(c.img)}" alt=""></div>`:`<div class="imgbox"><span class="noimg">no trusted photo<br>judge from the text</span></div>`;
  const btns=zone==="query"?"":`<div class="btns">${zone==="similar"?`<button data-act="up" title="more similar">▲</button><button data-act="down" title="less similar">▼</button>`:`<button data-act="similar" title="Similar">✓</button>`}${zone==="not"?"":`<button data-act="not" title="Not similar">✗</button>`}${zone==="tray"?"":`<button data-act="tray" title="Back to sort">↺</button>`}</div>`;
  return `<div class="card" ${zone==="query"?"":`draggable="true"`} data-key="${esc(c.key)}">${rank?`<span class="rank">${rank}</span>`:""}${img}
  <div class="cbody"><b>${esc(c.subfamily)}</b><span>${esc(c.desc)}</span><span>${esc(c.colour)}${c.finishing?" · "+esc(c.finishing):""}</span>
  <span class="small">${esc(c.material||c.composition)}</span><span><b>${esc(c.price)} €</b> <span class="small">${esc(c.key)}</span></span></div>${btns}</div>`;
}
const Q=()=>D.queries[cur];
function initPlace(){
  const q=Q(),a=D.answers[q.query];
  place=a?{tray:[],similar:[...a.similar],not:[...a.not]}:{tray:q.candidates.map(c=>c.key),similar:[],not:[]};
  started=Date.now();
}
function render(){
  const q=Q(),by=Object.fromEntries(q.candidates.map(c=>[c.key,c]));
  $("#qcard").innerHTML=cardHtml(q.item,"query");
  for(const z of ["tray","similar","not"])$("#"+z).innerHTML=place[z].map((k,i)=>cardHtml(by[k],z,z==="similar"?i+1:0)).join("")||`<span class="small">${z==="tray"?"All placed ✓":"Drop here"}</span>`;
  $("#left").textContent=place.tray.length?`${place.tray.length} left`:"";
  $("#save").disabled=place.tray.length>0;
  const done=D.queries.filter(x=>D.answers[x.query]).length;
  $("#bar").style.width=100*done/D.queries.length+"%";$("#ptext").textContent=`${done} / ${D.queries.length} queries · ${cur+1} shown`;
  $("#jump").innerHTML=D.queries.map((x,i)=>`<option value="${i}" ${i===cur?"selected":""}>${D.answers[x.query]?"✓":"·"} ${i+1}. ${esc(x.item.subfamily)} (${esc(x.query)})</option>`).join("");
  $("#prev").disabled=cur===0;$("#next").disabled=cur===D.queries.length-1;
  learning();
}
function learning(){
  const L=D.learning,max=Math.max(...L.blocks.map(b=>Math.max(L.start[b],L.learned[b])))*1.1;
  $("#weights").innerHTML=L.blocks.map(b=>{const s=L.start[b],l=L.learned[b],d=l-s;
    return `<div class="wrow"><span>${NAMES[b]||b}</span><div class="wbar"><div class="l" style="width:${100*l/max}%"></div><div class="s" style="width:${100*s/max}%"></div></div>
    <span class="${Math.abs(d)<.005?"":d>0?"up":"down"}">${(100*l).toFixed(0)}%</span></div>`}).join("");
  const h=L.holdout,pct=v=>v==null||isNaN(v)?"–":Math.round(100*v)+"%";
  $("#lstats").innerHTML=`Learned from ${L.pairs} preferences on ${L.queries} queries.<br>`+(h.learned.pairs?
   `Check on ${h.learned.pairs} preferences the engine never learned from: start ${pct(h.start.pair_accuracy)} → learned <b>${pct(h.learned.pair_accuracy)}</b> in the right order.`:
   "The check on unseen queries appears once some of them are sorted.");
}
function move(key,to,idx){
  for(const z of ["tray","similar","not"])place[z]=place[z].filter(k=>k!==key);
  if(idx==null||idx>place[to].length)place[to].push(key);else place[to].splice(idx,0,key);
  render();
}
async function load(){
  D=await (await fetch("/api/state")).json();
  $("#title").textContent=`Similarity feedback: ${D.name}`;document.title=`Feedback: ${D.name}`;
  cur=Math.max(0,D.queries.findIndex(x=>!D.answers[x.query]));if(!D.queries.length){$("#qcard").textContent="No queries for you in the pool.";return}
  initPlace();render();
}
function go(i){if(i<0||i>=D.queries.length)return;cur=i;initPlace();render();scrollTo(0,0)}
document.addEventListener("click",async e=>{
  const t=e.target,c=t.closest(".card");
  if(t.matches(".imgbox img")){const lb=$("#lightbox");lb.querySelector("img").src=t.src;lb.style.display="grid";return}
  if(t.closest("#lightbox")){$("#lightbox").style.display="none";return}
  if(t.dataset.act&&c){const k=c.dataset.key,a=t.dataset.act;
    if(a==="up"||a==="down"){const i=place.similar.indexOf(k),j=a==="up"?i-1:i+1;if(j<0||j>=place.similar.length)return;
      place.similar.splice(i,1);place.similar.splice(j,0,k);render();return}
    move(k,a);return}
  if(t.id==="prev")go(cur-1);if(t.id==="next")go(cur+1);
  if(t.id==="reset"){place={tray:Q().candidates.map(c=>c.key),similar:[],not:[]};render()}
  if(t.id==="save"){
    const r=await (await fetch("/api/save",{method:"POST",body:JSON.stringify({query:Q().query,similar:place.similar,not:place.not,seconds:(Date.now()-started)/1000})})).json();
    if(!r.ok){alert(r.error);return}
    D.answers[Q().query]={similar:[...place.similar],not:[...place.not]};D.learning=r.learning;toast("Saved · the engine learned from it");
    const nxt=D.queries.findIndex((x,i)=>i>cur&&!D.answers[x.query]),any=D.queries.findIndex(x=>!D.answers[x.query]);
    if(nxt>=0)go(nxt);else if(any>=0)go(any);else{render();toast("All your queries are done 🎉 Click “Submit my feedback”.")}}
  if(t.id==="submit"){
    $("#dlg").style.display="grid";$("#dlgMsg").textContent="Sending to GitHub…";
    const r=await (await fetch("/api/submit",{method:"POST",body:"{}"})).json();
    $("#dlgMsg").innerHTML=(r.ok?"<p>✅ Your feedback was sent to GitHub.</p>":`<p>⚠️ It could not be sent automatically. Send your file to the team instead: <b>${esc(D.file)}</b> (folder outputs/phase4).</p>`)+`<pre>${esc(r.log||r.error||"")}</pre>`}
  if(t.id==="closeDlg")$("#dlg").style.display="none";
});
$("#jump").addEventListener("change",e=>go(+e.target.value));
document.addEventListener("dragstart",e=>{const c=e.target.closest&&e.target.closest(".card[draggable]");if(!c)return;dragKey=c.dataset.key;c.classList.add("dragging");e.dataTransfer.effectAllowed="move"});
document.addEventListener("dragend",e=>{dragKey=null;document.querySelectorAll(".dragging,.over").forEach(x=>x.classList.remove("dragging","over"))});
document.querySelectorAll(".zone").forEach(z=>{
  z.addEventListener("dragover",e=>{e.preventDefault();z.classList.add("over")});
  z.addEventListener("dragleave",()=>z.classList.remove("over"));
  z.addEventListener("drop",e=>{e.preventDefault();z.classList.remove("over");if(!dragKey)return;
    const cards=[...z.querySelectorAll(".card")].filter(c=>c.dataset.key!==dragKey);
    let idx=cards.findIndex(c=>{const r=c.getBoundingClientRect();return e.clientY<r.top+r.height/2||(e.clientY<r.bottom&&e.clientX<r.left+r.width/2)});
    move(dragKey,z.dataset.zone,idx<0?null:idx)});
});
document.addEventListener("keydown",e=>{if(e.key==="Escape")$("#lightbox").style.display="none"});
load();
</script></body></html>"""

if __name__ == "__main__":
    main()
