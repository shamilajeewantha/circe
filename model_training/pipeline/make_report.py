"""Step 4 — build the visual analysis report.

Reads reports/stats.json + reports/embedding.json + reports/dropped_images.csv and
the merged/ output, renders figures to reports/figures/, and writes a SELF-CONTAINED
reports/index.html (PNGs embedded as base64) that can be published as an Artifact.

Runs in the base env (matplotlib / numpy / cv2 / PIL) — no FiftyOne needed:
    python pipeline/make_report.py
"""
import base64
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import MERGED_DIR, REPORTS_DIR, SPLIT_RATIOS, TAXONOMY

FIG = REPORTS_DIR / "figures"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def b64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


# ---------- figures ----------
def fig_class_balance(stats):
    totals = stats["class_totals"]
    vals = [totals[c] for c in TAXONOMY]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(TAXONOMY, vals, color="#4C78A8")
    ax.set_ylabel("boxes"); ax.set_title("Instances per class (merged)")
    ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.text(i, max(v, 1), str(v), ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    return save(fig, "class_balance.png")


def fig_composition(stats):
    src_cls = stats["per_source_class"]
    sources = sorted(src_cls)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(TAXONOMY))
    for i, src in enumerate(sources):
        vals = np.array([src_cls[src].get(c, 0) for c in TAXONOMY], dtype=float)
        ax.bar(TAXONOMY, vals, bottom=bottom, label=src, color=cmap(i % 20))
        bottom += vals
    ax.set_ylabel("boxes"); ax.set_title("Class composition by source")
    ax.legend(fontsize=6, ncol=2)
    plt.xticks(rotation=30, ha="right")
    return save(fig, "composition.png")


def fig_source_contribution(stats):
    src_tot = {s: sum(c.values()) for s, c in stats["per_source_class"].items()}
    items = sorted(src_tot.items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#72B7B2")
    ax.set_xlabel("boxes"); ax.set_title("Total boxes contributed per source")
    return save(fig, "source_contribution.png")


def fig_umap(emb, color_key, title, name):
    x, y = emb["x"], emb["y"]
    labels = emb[color_key]
    cats = sorted(set(labels))
    cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
    idx = {c: i for i, c in enumerate(cats)}
    colors = [cmap(idx[l] % 20) for l in labels]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, c=colors, s=3, alpha=0.5, linewidths=0)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap(idx[c] % 20)) for c in cats]
    ax.legend(handles, cats, fontsize=6, ncol=2, markerscale=1.5)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return save(fig, name)


def fig_dropped(stats):
    dropped = stats.get("dropped_boxes", {})
    if not dropped:
        return None
    items = sorted(dropped.items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(items))))
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#E45756")
    ax.set_xlabel("boxes dropped"); ax.set_title("Dropped boxes (class-level) by reason")
    return save(fig, "dropped_boxes.png")


def fig_mosaics():
    """One montage per class: sample merged/train images with that class's boxes drawn."""
    lbl_dir = MERGED_DIR / "train" / "labels"
    img_dir = MERGED_DIR / "train" / "images"
    if not lbl_dir.is_dir():
        return {}
    by_class = defaultdict(list)
    for lf in lbl_dir.iterdir():
        ids = {int(l.split()[0]) for l in lf.read_text().splitlines() if l.strip()}
        for cid in ids:
            by_class[cid].append(lf)
    rng = random.Random(7)
    out = {}
    for cid, name in enumerate(TAXONOMY):
        files = by_class.get(cid, [])
        if not files:
            continue
        pick = rng.sample(files, min(6, len(files)))
        tiles = []
        for lf in pick:
            imgs = list(img_dir.glob(lf.stem + ".*"))
            if not imgs:
                continue
            im = cv2.imread(str(imgs[0]))
            if im is None:
                continue
            h, w = im.shape[:2]
            for line in lf.read_text().splitlines():
                p = line.split()
                if len(p) < 5 or int(p[0]) != cid:
                    continue
                xc, yc, bw, bh = (float(v) for v in p[1:5])
                x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
            tiles.append(cv2.resize(im, (256, 256)))
        if not tiles:
            continue
        while len(tiles) < 6:
            tiles.append(np.full((256, 256, 3), 40, np.uint8))
        grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
        p = FIG / f"mosaic_{name}.png"
        FIG.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), grid)
        out[name] = p
    return out


def load_drop_summary():
    csv_path = REPORTS_DIR / "dropped_images.csv"
    if not csv_path.exists():
        return {}
    summ = Counter()
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            summ[f"{row['stage']} — {row['reason']}"] += 1
    return dict(summ)


# ---------- html ----------
def main():
    stats = json.loads((REPORTS_DIR / "stats.json").read_text(encoding="utf-8"))
    # embedding.json is written only by the embedding-based curation path (MCP / compute_visualization).
    # The fast exact-duplicate path skips it; the UMAP section is then omitted from the report.
    embp = REPORTS_DIR / "embedding.json"
    emb = json.loads(embp.read_text(encoding="utf-8")) if embp.exists() else None
    drop_summary = load_drop_summary()

    # duplicate detection was either embedding near-dup (emb present) or exact byte-identical (fast path)
    dup_label = "Near-duplicates removed" if emb is not None else "Exact duplicates removed"
    dup_word = "near-duplicate" if emb is not None else "exact-duplicate (byte-identical)"

    figs = {
        "balance": fig_class_balance(stats),
        "composition": fig_composition(stats),
        "sources": fig_source_contribution(stats),
    }
    if emb is not None:
        figs["umap_src"] = fig_umap(emb, "source", "UMAP embedding — colored by source", "umap_source.png")
        figs["umap_cls"] = fig_umap(emb, "primary", "UMAP embedding — colored by primary class", "umap_class.png")
    dropped = fig_dropped(stats)
    mosaics = fig_mosaics()

    totals = stats["class_totals"]
    nonzero = [v for v in totals.values() if v]
    imbalance = (max(nonzero) / min(nonzero)) if nonzero else 0

    # dynamic facts for the assessment
    per_src = stats["per_source_class"]
    class_sources = {c: [s for s, cc in per_src.items() if cc.get(c, 0) > 0] for c in TAXONOMY}
    populated = [c for c in TAXONOMY if totals[c] > 0]
    single_source = [c for c in populated if len(class_sources[c]) == 1]
    top_class = max(populated, key=lambda c: totals[c])
    low_class = min(populated, key=lambda c: totals[c])
    n_sources = len(per_src)

    def img(path, w="100%"):
        return f'<img src="{b64(path)}" alt="" style="max-width:{w};width:100%">'

    def tile(k, v, sub=""):
        return f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="sub">{sub}</div></div>'

    tiles = "".join([
        tile("Images", f"{stats['n_samples']:,}", f"{n_sources} sources"),
        tile("Labeled boxes", f"{sum(totals.values()):,}", f"{len(populated)} classes"),
        tile("Classes", f"{len(TAXONOMY)}", "detection taxonomy"),
        tile("Imbalance", f"{imbalance:.0f}×", f"{top_class} : {low_class}"),
        tile(dup_label, f"{stats['n_duplicates']:,}", "before splitting"),
        tile("Single-source classes", f"{len(single_source)}", "domain-risk"),
    ])
    rows = "".join(
        f"<tr><td class='mono'>{i}</td><td>{c}</td><td class='num'>{totals[c]:,}</td>"
        f"<td class='num'>{len(class_sources[c])}</td></tr>" for i, c in enumerate(TAXONOMY))
    drop_rows = "".join(f"<tr><td>{k}</td><td class='num'>{v:,}</td></tr>"
                        for k, v in sorted(drop_summary.items(), key=lambda x: -x[1])) or \
        "<tr><td>none</td><td class='num'>0</td></tr>"
    mosaic_html = "".join(f'<figure><figcaption>{name}</figcaption>{img(p)}</figure>' for name, p in mosaics.items())
    single_txt = ", ".join(f"<code>{c}</code>" for c in single_source) or "none"

    # UMAP domain map — only when the embedding path produced coordinates
    umap_section = ""
    if emb is not None:
        umap_section = f"""
<section>
<h2>Domain distribution (UMAP of image embeddings)</h2>
<p class="desc">Each point is one image, positioned by visual similarity. Points that separate by source
while sharing a class indicate a domain gap. Note that the fire and smoke data includes outdoor and
wildfire scenes; this is retained deliberately and documented rather than removed.</p>
<div class="two">
<figure><figcaption>Colored by source</figcaption>{img(figs['umap_src'])}</figure>
<figure><figcaption>Colored by primary class</figcaption>{img(figs['umap_cls'])}</figure>
</div>
</section>"""

    html = f"""<title>Circe inspection dataset — analysis &amp; merge report</title>
<style>
:root{{--bg:#F1F3F6;--surface:#FFF;--surface2:#F7F9FB;--ink:#191D23;--muted:#59626E;--line:#DDE2E9;
--accent:#B86A16;--kept:#2C7C6B;--drop:#AE4429;--radius:12px;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151A;--surface:#1B1F26;--surface2:#20252D;--ink:#E8ECF1;
--muted:#98A1AD;--line:#2B313A;--accent:#DE9433;--kept:#4FB4A1;--drop:#D66A4C;}}}}
:root[data-theme="dark"]{{--bg:#12151A;--surface:#1B1F26;--surface2:#20252D;--ink:#E8ECF1;--muted:#98A1AD;
--line:#2B313A;--accent:#DE9433;--kept:#4FB4A1;--drop:#D66A4C;}}
:root[data-theme="light"]{{--bg:#F1F3F6;--surface:#FFF;--surface2:#F7F9FB;--ink:#191D23;--muted:#59626E;
--line:#DDE2E9;--accent:#B86A16;--kept:#2C7C6B;--drop:#AE4429;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}}
.wrap{{max-width:1000px;margin:0 auto;padding:56px 24px 80px}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin:0 0 14px}}
h1{{font-size:clamp(28px,4vw,42px);line-height:1.1;letter-spacing:-.02em;margin:0 0 14px;text-wrap:balance}}
h2{{font-size:22px;letter-spacing:-.01em;margin:0 0 6px;border-bottom:1px solid var(--line);padding-bottom:10px}}
.lede{{color:var(--muted);font-size:17px;max-width:64ch;margin:0}}
section{{margin-top:52px}}p.desc{{color:var(--muted);font-size:15px;max-width:66ch}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-top:28px}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:18px}}
.tile .k{{font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}}
.tile .v{{font-family:var(--mono);font-size:26px;font-weight:600;margin-top:8px;font-variant-numeric:tabular-nums}}
.tile .sub{{font-size:12px;color:var(--muted);margin-top:2px}}
figure{{margin:22px 0;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:16px}}
figure img{{border-radius:6px;display:block}}
figcaption{{font-family:var(--mono);font-size:12px;color:var(--muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:.08em}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:720px){{.two{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;margin-top:16px;font-size:14px}}
th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left}}
th{{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}
td.num,.num{{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}}
td.mono{{font-family:var(--mono);color:var(--muted)}}
ul.rec{{list-style:none;padding:0;margin:20px 0 0;display:flex;flex-direction:column;gap:14px}}
ul.rec li{{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:10px;padding:14px 16px}}
ul.rec li.q{{border-left-color:var(--drop)}}
code{{font-family:var(--mono);font-size:.9em;background:var(--surface2);padding:1px 5px;border-radius:4px}}
footer{{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);font-family:var(--mono);font-size:12px;color:var(--muted)}}
</style>
<div class="wrap">
<p class="eyebrow">Factory-floor inspection rover · dataset audit</p>
<h1>Merged detection dataset — analysis &amp; merge report</h1>
<p class="lede">A consolidated {len(TAXONOMY)}-class YOLO detection dataset assembled from {n_sources}
source datasets for a ground inspection robot. Sources under <code>datasets/</code> are treated as
read-only; the pipeline copies pixels into <code>merged/</code> after de-duplication and a fresh
stratified split.</p>
<div class="tiles">{tiles}</div>

<section>
<h2>Class balance</h2>
<p class="desc">{sum(totals.values()):,} labeled instances across {len(populated)} populated classes. The
distribution is highly skewed (approximately {imbalance:.0f}× between the largest class,
<code>{top_class}</code>, and the smallest, <code>{low_class}</code>).</p>
<figure><figcaption>Instances per class (log scale)</figcaption>{img(figs['balance'])}</figure>
<table><tr><th>id</th><th>class</th><th>boxes</th><th>sources</th></tr>{rows}</table>
</section>

<section>
<h2>Composition by source</h2>
<p class="desc">Contribution of each source dataset to each class. Classes drawn from a single source
carry a higher risk of domain overfitting and are called out in the assessment below.</p>
<figure><figcaption>Class composition by source</figcaption>{img(figs['composition'])}</figure>
<figure><figcaption>Total boxes contributed per source</figcaption>{img(figs['sources'])}</figure>
</section>

{umap_section}

<section>
<h2>Data cleanup and dropped items</h2>
<p class="desc">Every dropped image is recorded with a reason in
<code>reports/dropped_images.csv</code>. Box-level drops (junk or out-of-taxonomy classes) are
summarized below. {stats['n_duplicates']:,} {dup_word} images were removed prior to splitting to
prevent train/validation leakage.</p>
{('<figure><figcaption>Dropped boxes by reason</figcaption>' + img(dropped) + '</figure>') if dropped else ''}
<table><tr><th>stage — reason</th><th>images</th></tr>{drop_rows}</table>
</section>

<section>
<h2>Representative samples per class</h2>
<p class="desc">Random samples from <code>merged/train</code> with ground-truth boxes drawn.</p>
{mosaic_html}
</section>

<section>
<h2>Assessment and recommendations</h2>
<ul class="rec">
<li class="q"><b>Class imbalance (~{imbalance:.0f}×).</b> <code>{top_class}</code> dominates while
<code>{low_class}</code> is sparse. Apply class weighting or targeted oversampling during training, and
evaluate per-class recall — the minority classes will otherwise underperform.</li>
<li class="q"><b>Single-source classes ({len(single_source)}).</b> {single_txt} are each drawn from a
single dataset and may overfit that source's domain. Prioritize additional, in-domain imagery for
these before deployment.</li>
<li><b>Label quality.</b> <code>fluid-patch__welding-defects</code> contains at least one confirmed
mislabel (a mouse pad annotated as oil); a targeted review of that source is recommended. Junk classes
(<code>Other</code>, <code>object</code>) and gauge needle/tick keypoints were removed during import.</li>
<li><b>Fire / smoke domain.</b> Training data spans indoor and outdoor (including wildfire) scenes. This
is intentional and noted here for transparency; no pruning was applied.</li>
</ul>
</section>

<footer>Read-only sources in datasets/ → FiftyOne curation (dedup + stratified split) → merged/ (YOLO,
{len(TAXONOMY)} classes) → YOLO26 training in the WSL yolo_det environment.</footer>
</div>"""
    (REPORTS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {REPORTS_DIR/'index.html'} ({len(figs)+ (1 if dropped else 0) + len(mosaics)} figures)")


if __name__ == "__main__":
    main()
