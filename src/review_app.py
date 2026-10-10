"""Phase 1 review app: review your batch in the browser, no CSV editing needed.

    python src/review_app.py --id 4          # Manuel (see TEAM in config.py)

Opens http://localhost:8765 in the browser. Every click is saved straight into your
batch CSV (outputs/phase1/batches/batch_0K_of_05.csv), in the same format the merge
step reads. Only the Python standard library and pandas are used. Stop with Ctrl+C or
by closing the window.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BATCH_DIR, EMB_DIR, IMG_DIR, ISSUES, KEY, PROCESSED_DIR,  # noqa: E402
                                REVIEW_STATUSES, ROOT, SEVERITY_RANK, TEAM)

N_MEMBERS = len(TEAM)

# Plain-language explanation of each automatic flag, shown on the cards.
ISSUE_LABELS = {
    "IMG_PATH_INVALID": "The image path is broken",
    "IMG_FILE_MISSING": "No photo for this product",
    "IMG_UNREADABLE": "The photo file is damaged",
    "IMG_REF_MISMATCH": "The photo file belongs to another product",
    "IMG_CAT_MISMATCH": "The photo is filed under another category",
    "IMG_COLOUR_MISMATCH": "The photo file is for another colour",
    "IMG_GENERIC": "The same photo is used for every colour: check the colour matches",
    "IMG_SHARED": "This photo is also used by another product",
    "TAB_KEY_INCONSISTENT": "Product code and colour code don't match",
    "CLR_CODE_NAME_INCONSISTENT": "Colour name differs from other products with the same colour code",
    "CLR_CONFLICT_DESC": "The description mentions a different colour",
    "CLR_NOT_IN_DESC": "The description doesn't mention the colour",
    "TYPE_CONFLICT_DESC": "The description names a different product type than the family",
    "SALES_MISSING": "No sales data",
    "SALES_ZERO_QTY": "Zero units sold",
    "SKU_ATTR_CONFLICT": "Different sizes have different values",
}
# Fields a reviewer may correct (column -> label shown in the form).
EDITABLE = {
    "PROD_DES_BASE": "Description",
    "CAT_DES_EN": "Category",
    "GFA_DES_EN": "Family",
    "GFS_DES_EN": "Sub-family",
    "CLR_DES": "Colour",
    "COMPOSITION": "Composition",
    "FINISHING": "Finishing",
    "MATERIAL": "Material",
}


class Batch:
    """The member's batch CSV, kept in memory and written back on every change."""

    def __init__(self, member: int):
        self.member = member
        self.name = TEAM[member]
        self.path = BATCH_DIR / f"batch_{member:02d}_of_{N_MEMBERS:02d}.csv"
        if not self.path.exists():
            raise SystemExit(f"{self.path} not found. Run: python src/phase1_checks.py "
                             f"batch --members {N_MEMBERS} --id {member}")
        self.df = pd.read_csv(self.path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        self.lock = threading.Lock()
        self.extra = self._extra_context()
        self.vision = self._vision_flags()
        self.suggestions = self._suggestions()

    def _vision_flags(self) -> dict:
        """Phase 1b image flags (committed in data/embeddings), so nobody has to rerun the checks."""
        p = EMB_DIR / "image_audit.parquet"
        if not p.exists():
            return {}
        d = pd.read_parquet(p, columns=[KEY, "vis_issues", "vis_note"])
        d = d[d[KEY].isin(self.df[KEY]) & (d["vis_issues"] != "")]
        return {r[KEY]: (r["vis_issues"], r["vis_note"]) for r in d.to_dict("records")}

    def _extra_context(self) -> dict:
        """Details from the full check output (shared-with list, sizes) if available."""
        p = PROCESSED_DIR / "items_checked.parquet"
        if not p.exists():
            return {}
        cols = [KEY, "img_shared_with", "sizes", "SALES_QTY"]
        d = pd.read_parquet(p, columns=[c for c in cols if c != KEY] + [KEY])
        d = d[d[KEY].isin(self.df[KEY])]
        return {r[KEY]: {c: ("" if pd.isna(r[c]) else str(r[c])) for c in cols if c != KEY}
                for r in d.to_dict("records")}

    def _suggestions(self) -> dict:
        p = PROCESSED_DIR / "items_checked.parquet"
        src = pd.read_parquet(p, columns=list(EDITABLE)) if p.exists() else self.df
        return {c: sorted(src[c].dropna().astype(str).unique().tolist())[:2000]
                for c in EDITABLE if c in src.columns and c != "PROD_DES_BASE"}

    def items(self) -> list[dict]:
        out = []
        for r in self.df.to_dict("records"):
            codes = [c for c in r["issues"].split(";") if c]
            # Image-check codes (VIS_*, ...) have no label here: their plain-language note is added below.
            r["issue_labels"] = [ISSUE_LABELS[c] for c in codes if c in ISSUE_LABELS]
            r.update(self.extra.get(r[KEY], {}))
            if r[KEY] in self.vision:  # add image-check flags and raise the priority if needed
                vis_codes, note = self.vision[r[KEY]]
                r["issue_labels"] += [n[:1].upper() + n[1:] for n in note.split(" · ") if n]
                ranks = [SEVERITY_RANK[ISSUES[c]] for c in codes + vis_codes.split(";") if c in ISSUES]
                r["priority"] = {3: "high", 2: "medium", 1: "low", 0: "info"}.get(max(ranks, default=-1), "none")
            out.append(r)
        return out

    def save(self, key: str, status: str, fixes: str, notes: str) -> None:
        if status not in REVIEW_STATUSES | {""}:
            raise ValueError(f"invalid status {status!r}")
        with self.lock:
            idx = self.df.index[self.df[KEY] == key]
            if len(idx) != 1:
                raise ValueError(f"unknown product {key!r}")
            i = idx[0]
            self.df.at[i, "review_status"] = status
            self.df.at[i, "fixes"] = fixes if status == "fix" else ""
            self.df.at[i, "notes"] = notes
            self.df.at[i, "reviewer"] = self.name if (status or notes) else ""
            self._write()

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        self.df.to_csv(tmp, index=False, encoding="utf-8-sig")
        try:
            os.replace(tmp, self.path)
        except PermissionError as err:
            tmp.unlink(missing_ok=True)
            raise PermissionError(f"Can't save: {self.path.name} is open in another program (Excel?). Close it.") from err

    def submit(self) -> tuple[bool, str]:
        """git add/commit/push this member's CSV only."""
        if not shutil.which("git"):
            return False, (f"Git is not installed. Send the file {self.path.name} "
                           f"(folder outputs/phase1/batches) to the team instead.")
        rel = self.path.relative_to(ROOT).as_posix()
        log = []
        steps = [["git", "add", rel],
                 ["git", "commit", "-m", f"Phase 1 review: batch {self.member} ({self.name})", "--", rel],
                 ["git", "pull", "--rebase", "--autostash"],
                 ["git", "push"]]
        for cmd in steps:
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            log.append(f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}".strip())
            nothing_new = cmd[1] == "commit" and "nothing" in (p.stdout + p.stderr).lower()
            if p.returncode != 0 and not nothing_new:
                return False, "\n\n".join(log)
        return True, "\n\n".join(log)


def make_handler(batch: Batch):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/items":
                self._json({"member": batch.member, "name": batch.name, "file": batch.path.name,
                            "editable": EDITABLE, "suggestions": batch.suggestions, "items": batch.items()})
            elif path.startswith("/img/"):
                name = Path(unquote(path[5:])).name  # file name only: no directory traversal
                f = IMG_DIR / name
                if not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = "image/png" if f.suffix.lower() == ".png" else "image/jpeg"
                self._send(200, f.read_bytes(), ctype)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if path == "/api/save":
                    batch.save(body["key"], body.get("status", ""), body.get("fixes", ""), body.get("notes", ""))
                    self._json({"ok": True})
                elif path == "/api/submit":
                    ok, log = batch.submit()
                    self._json({"ok": ok, "log": log})
                else:
                    self._json({"ok": False, "error": "unknown endpoint"}, 404)
            except Exception as e:  # report to the page instead of crashing the server
                self._json({"ok": False, "error": str(e)}, 400)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, required=True, help="your team number: " +
                    ", ".join(f"{k}={v}" for k, v in TEAM.items()))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if args.id not in TEAM:
        raise SystemExit(f"--id must be one of {list(TEAM)}")

    batch = Batch(args.id)
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(batch))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port found.")
    url = f"http://localhost:{server.server_port}"
    print(f"Review app for {batch.name} (batch {args.id}) running at {url}")
    print("Your decisions are saved automatically. Keep this window open while reviewing; close it when done.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parfois review</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--accent:#1f5f8b;
--ok:#2e7d32;--warn:#e08a00;--bad:#c62828;--info:#6d4c9f}
*{box-sizing:border-box}body{margin:0;font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
h1{font-size:17px;margin:0}.progress{flex:1;min-width:220px}.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar>div{height:100%;background:var(--ok);width:0}.small{font-size:12px;color:var(--muted)}
select,input,textarea,button{font:inherit}select,input,textarea{border:1px solid var(--line);border-radius:6px;padding:5px 7px;background:#fff;color:var(--ink)}
button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.help{background:#eef4f8;border-bottom:1px solid var(--line);padding:8px 16px;font-size:13px}
main{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.card.flag-high{border:2px solid var(--bad)}.card.flag-medium{border:2px solid var(--warn)}
.card.done{opacity:.72}.imgbox{aspect-ratio:1;background:#fafafa;display:grid;place-items:center;cursor:zoom-in}
.imgbox img{max-width:100%;max-height:100%;object-fit:contain}.noimg{color:var(--muted)}
.body{padding:10px;display:flex;flex-direction:column;gap:6px;flex:1}.code{font-weight:600}
.desc{font-size:15px}.meta{font-size:12px;color:var(--muted)}.flags{margin:0;padding-left:18px;font-size:12px;color:var(--bad)}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:0 10px 10px}
.actions button.sel{color:#fff}.b-ok.sel{background:var(--ok);border-color:var(--ok)}.b-drop_image.sel,.b-discard.sel{background:var(--bad);border-color:var(--bad)}
.b-fix.sel{background:var(--accent);border-color:var(--accent)}.b-team_review.sel{background:var(--info);border-color:var(--info)}
.status{font-size:12px;font-weight:600}.fixbox{display:none;padding:0 10px 10px;gap:6px;flex-direction:column}.fixbox.open{display:flex}
.fixrow{display:grid;grid-template-columns:90px 1fr;gap:6px;align-items:center;font-size:12px}
.pager{grid-column:1/-1;display:flex;gap:8px;justify-content:center;align-items:center}
#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;place-items:center;z-index:10;cursor:zoom-out}
#lightbox img{max-width:94vw;max-height:94vh;background:#fff}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1d1d1b;color:#fff;padding:8px 14px;border-radius:8px;display:none;z-index:11}
pre{white-space:pre-wrap;font-size:12px;max-height:300px;overflow:auto;background:#f0f0ec;padding:8px;border-radius:6px}
#submitDlg{position:fixed;inset:0;background:rgba(0,0,0,.4);display:none;place-items:center;z-index:12}
#submitDlg>div{background:#fff;border-radius:10px;padding:16px;max-width:640px;width:92vw}
@media (max-width:600px){main{padding:10px}header{padding:8px 10px}}
</style></head><body>
<header>
  <h1 id="title">Parfois review</h1>
  <div class="progress"><div class="bar"><div id="bar"></div></div><div class="small" id="progressText"></div></div>
  <label class="small">Show <select id="filter">
    <option value="todo">To check (flagged, not done)</option><option value="flagged">All flagged</option>
    <option value="all">All products</option><option value="done">Done</option><option value="problems">Marked as problem</option></select></label>
  <label class="small">Category <select id="cat"><option value="">All</option></select></label>
  <input id="search" placeholder="Search code or text" size="16">
  <button class="primary" id="submitBtn">Submit my review</button>
</header>
<div class="help">For each product: does the <b>photo</b>, the <b>description</b> and the <b>data</b> (category, family, colour) all describe the same article?
Click <b>Looks right</b>, <b>Wrong photo</b>, <b>Fix data</b>, <b>Not sure</b> or <b>Discard</b>. Everything saves automatically.
Products with no flag that you don't touch count as correct. Click a photo to enlarge it.</div>
<main id="grid"></main>
<div id="lightbox"><img alt=""></div><div id="toast"></div>
<div id="submitDlg"><div><h3 style="margin-top:0">Submit my review</h3><div id="submitMsg">Sending…</div>
<p><button id="closeDlg">Close</button></p></div></div>
<script>
const PAGE=60; let D, page=0;
const STATUS={ok:"Looks right",drop_image:"Wrong photo",fix:"Fix data",team_review:"Not sure",discard:"Discard"};
const FIELD={CLR_COD:"colour code",CLR_DES:"colour",CLR_TYPE:"colour type",CAT_DES_EN:"category",GFA_DES_EN:"family",
GFS_DES_EN:"sub-family",CATEGORY_MATRIX:"planning category",COMPOSITION:"composition",MATERIAL:"material",FINISHING:"finishing",
PRINT_TYPE:"print",OUTFIT:"outfit",DIMENSION:"size type",NUMBER_OF_UNITS:"units",THEME:"theme",FASHIONTYPE:"fashion type",
PROD_SEG:"segment",PRICE_BASE_W_VAT:"price",PROG_IMAGE:"photo"};
const $=s=>document.querySelector(s);
function toast(t){const e=$("#toast");e.textContent=t;e.style.display="block";clearTimeout(e._t);e._t=setTimeout(()=>e.style.display="none",1800)}
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function flagged(it){return it.priority==="high"||it.priority==="medium"||it.priority==="low"}
function parseFixes(s){const o={};(s||"").split(/;\s*(?=[A-Za-z_][A-Za-z0-9_]*\s*=)/).forEach(p=>{const i=p.indexOf("=");if(i>0)o[p.slice(0,i).trim()]=p.slice(i+1).trim()});return o}
function fixString(o){return Object.entries(o).filter(([k,v])=>v!=="").map(([k,v])=>k+"="+v).join("; ")}
async function load(){
  D=await (await fetch("/api/items")).json();
  $("#title").textContent=`Batch ${D.member}: ${D.name}`; document.title=`Review: ${D.name}`;
  [...new Set(D.items.map(i=>i.CAT_DES_EN))].sort().forEach(c=>$("#cat").insertAdjacentHTML("beforeend",`<option>${esc(c)}</option>`));
  for(const [col,vals] of Object.entries(D.suggestions)){const dl=document.createElement("datalist");dl.id="dl_"+col;dl.innerHTML=vals.map(v=>`<option value="${esc(v)}">`).join("");document.body.appendChild(dl)}
  render();
}
function visible(){
  const f=$("#filter").value,c=$("#cat").value,q=$("#search").value.trim().toLowerCase();
  return D.items.filter(it=>{
    if(c&&it.CAT_DES_EN!==c)return false;
    if(q&&!(it.PROD_CLR_EQUIV+" "+it.PROD_DES_BASE+" "+it.GFA_DES_EN).toLowerCase().includes(q))return false;
    if(f==="todo")return flagged(it)&&!it.review_status;
    if(f==="flagged")return flagged(it);
    if(f==="done")return !!it.review_status;
    if(f==="problems")return ["fix","drop_image","discard","team_review"].includes(it.review_status);
    return true});
}
function progress(){
  const fl=D.items.filter(flagged),fd=fl.filter(i=>i.review_status).length,done=D.items.filter(i=>i.review_status).length;
  $("#bar").style.width=(fl.length?100*fd/fl.length:100)+"%";
  $("#progressText").textContent=`Flagged products checked: ${fd} / ${fl.length} · decisions on all products: ${done} / ${D.items.length}`;
}
function card(it){
  const img=it.img_file?`<div class="imgbox" data-img="/img/${encodeURIComponent(it.img_file)}"><img loading="lazy" src="/img/${encodeURIComponent(it.img_file)}" alt=""></div>`:`<div class="imgbox"><span class="noimg">No photo</span></div>`;
  const flags=it.issue_labels.filter(l=>l!=="No sales data").map(l=>{
    if(l.startsWith("This photo is also used")&&it.img_shared_with)return `${esc(l)}: ${esc(it.img_shared_with.split("|").filter(k=>k!==it.PROD_CLR_EQUIV).join(", "))}`;
    if(l.startsWith("Different sizes")&&it.sku_conflicts)return `${esc(l)}: ${esc(it.sku_conflicts.split("|").map(c=>FIELD[c]||c).join(", "))}`;
    return esc(l)}).map(l=>`<li>${l}</li>`).join("");
  const fx=parseFixes(it.fixes);
  const rows=Object.entries(D.editable).map(([col,label])=>`<div class="fixrow"><span>${label}</span><input data-col="${col}" list="dl_${col}" value="${esc(fx[col]??"")}" placeholder="${esc(it[col]||"(empty)")}"></div>`).join("");
  const st=it.review_status;
  return `<div class="card ${flagged(it)?"flag-"+it.priority:""} ${st?"done":""}" data-key="${esc(it.PROD_CLR_EQUIV)}">
  ${img}<div class="body"><div class="code">${esc(it.PROD_CLR_EQUIV)} <span class="small">· €${esc(it.PRICE_BASE_W_VAT)}</span></div>
  <div class="desc">${esc(it.PROD_DES_BASE)}</div>
  <div class="meta">${esc(it.CAT_DES_EN)} › <b>${esc(it.GFA_DES_EN)}</b> › ${esc(it.GFS_DES_EN)}<br>Colour: <b>${esc(it.CLR_DES)}</b>${it.FINISHING?" · Finish: "+esc(it.FINISHING):""}<br>${esc(it.COMPOSITION)}</div>
  ${flags?`<ul class="flags">${flags}</ul>`:""}
  <div class="status">${st?"✓ "+STATUS[st]:""}</div></div>
  <div class="actions">${Object.entries(STATUS).map(([k,v])=>`<button class="b-${k} ${st===k?"sel":""}" data-status="${k}">${v}</button>`).join("")}
  <input class="note" placeholder="Note (optional)" value="${esc(it.notes)}" style="grid-column:span 1"></div>
  <div class="fixbox ${st==="fix"?"open":""}"><div class="small">Type only the values that are wrong (suggestions appear as you type):</div>${rows}<button class="primary saveFix">Save fix</button></div></div>`;
}
function render(){
  const v=visible(),pages=Math.max(1,Math.ceil(v.length/PAGE));page=Math.min(page,pages-1);
  const slice=v.slice(page*PAGE,(page+1)*PAGE);
  const pager=`<div class="pager"><button data-p="-1" ${page?"":"disabled"}>‹ Previous</button><span class="small">Page ${page+1} of ${pages} · ${v.length} products</span><button data-p="1" ${page<pages-1?"":"disabled"}>Next ›</button>
  ${$("#filter").value==="all"?`<button id="okPage">Mark untouched on this page as “Looks right”</button>`:""}</div>`;
  $("#grid").innerHTML=(slice.length?slice.map(card).join(""):`<p class="small">Nothing to show here. ${$("#filter").value==="todo"?"All flagged products are checked 🎉. Now skim the rest with “All products”, then click “Submit my review”.":""}</p>`)+pager;
  progress();
}
async function save(it,status,fixes,notes){
  const r=await (await fetch("/api/save",{method:"POST",body:JSON.stringify({key:it.PROD_CLR_EQUIV,status,fixes,notes})})).json();
  if(!r.ok){alert(r.error);return false}
  Object.assign(it,{review_status:status,fixes:status==="fix"?fixes:"",notes});toast("Saved");return true;
}
const byKey=k=>D.items.find(i=>i.PROD_CLR_EQUIV===k);
document.addEventListener("click",async e=>{
  const t=e.target, c=t.closest(".card"), it=c&&byKey(c.dataset.key);
  if(t.closest(".imgbox[data-img]")){const lb=$("#lightbox");lb.querySelector("img").src=t.closest(".imgbox").dataset.img;lb.style.display="grid";return}
  if(t.closest("#lightbox")){$("#lightbox").style.display="none";return}
  if(t.dataset.p){page+=+t.dataset.p;render();scrollTo(0,0);return}
  if(t.id==="okPage"){const v=visible().slice(page*PAGE,(page+1)*PAGE).filter(i=>!i.review_status);for(const i of v)await save(i,"ok","",i.notes);render();return}
  if(t.dataset.status&&it){
    const s=t.dataset.status,note=c.querySelector(".note").value;
    if(s==="fix"){c.querySelector(".fixbox").classList.add("open");return}
    const newS=it.review_status===s?"":s;  // click again to undo
    if(await save(it,newS,"",note))c.outerHTML=card(it),progress();return}
  if(t.classList.contains("saveFix")&&it){
    const o={};c.querySelectorAll(".fixbox input").forEach(i=>{if(i.value.trim())o[i.dataset.col]=i.value.trim()});
    if(!Object.keys(o).length){alert("Type at least one corrected value.");return}
    if(await save(it,"fix",fixString(o),c.querySelector(".note").value))c.outerHTML=card(it),progress();return}
  if(t.id==="submitBtn"){
    const fl=D.items.filter(flagged),left=fl.filter(i=>!i.review_status).length;
    if(left&&!confirm(`${left} flagged products are still unchecked. Submit anyway? (You can submit again later.)`))return;
    $("#submitDlg").style.display="grid";$("#submitMsg").textContent="Sending to GitHub…";
    const r=await (await fetch("/api/submit",{method:"POST",body:"{}"})).json();
    $("#submitMsg").innerHTML=(r.ok?"<p>✅ Your review was sent to GitHub.</p>":"<p>⚠️ It could not be sent automatically. Send your file to the team instead: <b>"+esc(D.file)+"</b> (folder outputs/phase1/batches).</p>")+`<pre>${esc(r.log||r.error||"")}</pre>`;return}
  if(t.id==="closeDlg"){$("#submitDlg").style.display="none"}
});
document.addEventListener("change",async e=>{
  const t=e.target;
  if(t.classList.contains("note")){const c=t.closest(".card"),it=byKey(c.dataset.key);await save(it,it.review_status||"",it.fixes,t.value);return}
  if(["filter","cat"].includes(t.id)){page=0;render()}
});
$("#search").addEventListener("input",()=>{page=0;render()});
load();
</script></body></html>"""

if __name__ == "__main__":
    main()
