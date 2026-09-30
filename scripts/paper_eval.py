#!/usr/bin/env python3
import argparse
import glob
import gzip
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import trimesh
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from src.metrics import ChamferDistance, FScore, NormalConsistency
from src.model.shape.VAE.postprocess import Postprocess

# The bank is built as [near sigma=.01 | near sigma=.05 | uniform]; the uniform block is the tail.
UNIFORM_BLOCK = slice(200_000, 250_000)
SURF_SAMPLES = 100_000    # points sampled off each mesh for chamfer / f-score / NC
SIOU_TAU = 0.02           # near-surface band for S-IoU, matching src/metrics/iou.py's default
DECODE_CHUNK = 50_000     # 250k query points x 1024 latents at once OOMs the decoder cross-attn


def fingerprint(args):
    """Everything that changes a per-shape number. Stored in each sidecar record and checked on
    resume: without it, switching --sample-posterior (or the MC resolution, or the split) would
    silently reuse results computed under the old setting and produce a table that is half one
    protocol and half another."""
    keys = ("split", "seed", "resolution", "fscore_tau", "no_mesh", "sample_posterior")
    return "|".join(f"{k}={getattr(args, k)}" for k in keys)


def load_run(ckpt_path, device):
    """A checkpoint carries its own cfg, but the run dir's .hydra copy is the resolved one."""
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    hydra_cfg = os.path.join(run_dir, ".hydra", "config.yaml")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(hydra_cfg) if os.path.isfile(hydra_cfg) else OmegaConf.create(sd["cfg"])
    model = instantiate(cfg.model)
    model.load_state_dict(sd["model"])
    # lambda-VAE's schedule buffers are persistent=False (so pre-lambda checkpoints stay loadable),
    # which also keeps them OUT of sd["model"] -- the runner saves them under "lam_state" instead.
    # Without restoring them, _lam() sees lam_star=1 at eval and injects sigma rather than
    # sigma^lambda: 2.6x the noise the decoder was trained on, which collapsed V-IoU from ~83 to 15.
    for k, v in (sd.get("lam_state") or {}).items():
        if hasattr(model, k):
            getattr(model, k).copy_(v)
    if float(getattr(model, "lam_delta", 0.0)) > 1.0 and not sd.get("lam_state"):
        raise SystemExit(f"{ckpt_path}: lam_delta={model.lam_delta} but the checkpoint has no "
                         f"lam_state; evaluating it would silently use lambda=1")
    model.eval().to(device)
    return model, instantiate(cfg.preprocess), cfg, sd.get("epoch")


def sample_fn(pre, cfg):
    """Whichever query-selection the run trained with -- a disjoint run evaluated on the plain
    sampler is fed an anchor distribution its encoder never saw."""
    return pre.sample_disjoint if "Disjoint" in str(cfg.data.train.dataset._target_) else pre.sample


def load_mesh(path):
    """Mirrors Preprocessor.build_cache: the dataset ships gzipped .off, and trimesh needs the
    stream already decompressed or it parses the gzip header as OFF text."""
    if path.endswith(".gz"):
        inner = os.path.splitext(path[:-3])[1].lstrip(".")
        with gzip.open(path, "rb") as f:
            return trimesh.load(f, file_type=inner, force="mesh")
    return trimesh.load_mesh(path, force="mesh")


def surface_points(mesh, n):
    pts, fi = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts), np.asarray(mesh.face_normals[fi])


def iou_counts(pred, gt, band=None):
    """Intersection/union COUNTS rather than a ratio.

    Counts are what make this resumable and memory-flat: pooled IoU is sum(inter)/sum(union) over
    shapes and per-shape IoU is mean(inter_i/union_i), so both aggregations fall out of the same
    two integers per shape -- no need to hold every shape's 250k predictions to the end.
    """
    pi, gi = pred < 0, gt < 0
    if band is not None:
        m = gt.abs() < band
        pi, gi = pi & m, gi & m
        valid = m.any().item()
    else:
        valid = True
    inter = int((pi & gi).sum())
    union = int((pi | gi).sum())
    return {"inter": inter, "union": union, "valid": bool(valid and union > 0)}


def eval_shape(model, pre, post, smp, cache, mesh_path, args):
    """All metrics for ONE shape. Returns a flat json-serialisable dict."""
    torch.manual_seed(args.seed)                 # identical anchors AND identical eps per shape
    q, d, *_ = smp(cache)
    with torch.no_grad():
        mu, _ = model.encode(q[None].to(args.device), d[None].to(args.device),
                             sample_posterior=args.sample_posterior)

    qp_all, gt_all = cache["sdf_query_points"], cache["gt_sdf"]
    rec = {}
    for tag, sel in (("mix", slice(None)), ("uni", UNIFORM_BLOCK)):
        qp, gt = qp_all[sel], gt_all[sel]
        preds = []
        for i in range(0, qp.shape[0], DECODE_CHUNK):
            pe = pre.fourier_embedder(qp[i:i + DECODE_CHUNK])[None].to(args.device)
            with torch.no_grad():
                preds.append(model.decode(mu, pe)[0][0].cpu())
        pred = torch.cat(preds)
        rec[f"viou_{tag}"] = iou_counts(pred, gt)
        if tag == "mix":
            rec["siou_mix"] = iou_counts(pred, gt, band=SIOU_TAU)

    rec["no_surface"] = False
    if not args.no_mesh:
        # Only the plain CrossAttentionDecoder exposes the split API; MRLDecoder/AnchorDecoder
        # take extra arguments, so they fall back to the per-chunk path.
        fast_dec = model.decoder if type(model.decoder).__name__ == "CrossAttentionDecoder" else None
        recon = post.to_mesh(model.decode, mu, decoder=fast_dec)
        if recon is None:
            rec["no_surface"] = True          # no zero crossing -> mesh metrics uncomputable
        else:
            gt_mesh = pre.mesh2query.normalize_mesh(load_mesh(mesh_path), 0.98)
            rp, rn = surface_points(recon, SURF_SAMPLES)
            gp, gn = surface_points(gt_mesh, SURF_SAMPLES)
            rec["chamfer"] = ChamferDistance()(rp, gp)
            rec["fscore"] = FScore(tau=args.fscore_tau)(rp, gp)
            rec["nc"] = NormalConsistency()(rp, rn, gp, gn)
    return rec


def aggregate(recs):
    """Per-shape records -> the numbers that go in the table."""
    def pooled(key):
        i = sum(r[key]["inter"] for r in recs if r[key]["valid"])
        u = sum(r[key]["union"] for r in recs if r[key]["valid"])
        return 100.0 * i / u if u else float("nan")

    def per_shape(key):
        v = [r[key]["inter"] / r[key]["union"] for r in recs if r[key]["valid"]]
        return 100.0 * float(np.mean(v)) if v else float("nan")

    def mesh(key):
        v = [r[key] for r in recs if key in r and not r["no_surface"]]
        return float(np.mean(v)) if v else float("nan")

    return {
        "shapes": len(recs),
        "viou_mix_pooled": pooled("viou_mix"), "viou_mix_pershape": per_shape("viou_mix"),
        "siou_mix_pooled": pooled("siou_mix"), "siou_mix_pershape": per_shape("siou_mix"),
        "viou_uni_pooled": pooled("viou_uni"), "viou_uni_pershape": per_shape("viou_uni"),
        "chamfer": mesh("chamfer"), "fscore": mesh("fscore"), "nc": mesh("nc"),
        "no_surface": sum(r["no_surface"] for r in recs),
    }


def render(results, order, args):
    """One renderer for both the terminal and the markdown file, so they can never disagree."""
    rows = [("epoch", "epoch", "{:.0f}"), ("shapes evaluated", "shapes", "{:.0f}")]
    sec_mix = [("V-IoU  pooled", "viou_mix_pooled", "{:.2f}"),
               ("V-IoU  per-shape", "viou_mix_pershape", "{:.2f}"),
               ("S-IoU  pooled", "siou_mix_pooled", "{:.2f}"),
               ("S-IoU  per-shape", "siou_mix_pershape", "{:.2f}")]
    sec_uni = [("V-IoU  pooled", "viou_uni_pooled", "{:.2f}"),
               ("V-IoU  per-shape", "viou_uni_pershape", "{:.2f}")]
    sec_mesh = [("Chamfer-L1 (lower better)", "chamfer", "{:.5f}"),
                (f"F-score@{args.fscore_tau:g}", "fscore", "{:.4f}"),
                ("Normal consistency", "nc", "{:.4f}"),
                ("shapes with no surface", "no_surface", "{:.0f}")]

    txt, md = [], []
    w = 36 + 14 * len(order)
    txt.append("=" * w)
    txt.append(f"  {'':34s}" + "".join(f"{c:>14s}" for c in order))
    txt.append("=" * w)
    md.append(f"# Reconstruction evaluation\n")
    md.append(f"- generated: {datetime.now():%Y-%m-%d %H:%M}")
    md.append(f"- split: `data/{args.split}`, {results[order[0]]['shapes']} shapes, seed {args.seed}")
    md.append(f"- marching cubes resolution {args.resolution}, F-score tau {args.fscore_tau}")
    md.append(f"- V-IoU/S-IoU on the full 250k-point SDF bank per shape")
    md.append(f"- latent: {'z ~ q(z|x), one draw per shape (matches training)' if args.sample_posterior else 'z = mu (posterior mode)'}\n")
    md.append("| metric | " + " | ".join(order) + " |")
    md.append("|---|" + "---|" * len(order))

    def emit(label, key, fmt):
        txt.append(f"  {label:34s}" + "".join(f"{fmt.format(results[c][key]):>14s}" for c in order))
        md.append(f"| {label} | " + " | ".join(fmt.format(results[c][key]) for c in order) + " |")

    for l, k, f in rows:
        emit(l, k, f)
    txt.append("\n  -- mixed bank (near+uniform, this project's protocol) --")
    md.append("| **mixed bank (near+uniform)** | " + " | ".join("" for _ in order) + " |")
    for l, k, f in sec_mix:
        emit(l, k, f)
    txt.append("\n  -- uniform block only (3DShape2VecSet protocol) --")
    md.append("| **uniform block only (3DShape2VecSet)** | " + " | ".join("" for _ in order) + " |")
    for l, k, f in sec_uni:
        emit(l, k, f)
    if not args.no_mesh:
        txt.append(f"\n  -- marching cubes @ {args.resolution}, tau={args.fscore_tau} --")
        md.append(f"| **marching cubes @ {args.resolution}** | " + " | ".join("" for _ in order) + " |")
        for l, k, f in sec_mesh:
            emit(l, k, f)
    txt.append("=" * w)
    md.append("\n## checkpoints\n")
    for c in order:
        md.append(f"- **{c}** — `{results[c]['ckpt']}` (epoch {results[c]['epoch']})")
    return "\n".join(txt), "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable; NAME labels the column")
    ap.add_argument("--shapes", type=int, default=64, help="0 = the whole split")
    ap.add_argument("--shapes-file", default=None,
                    help="newline-separated split-relative shape ids (no extension). Overrides "
                         "--shapes/--seed, so a stratified subset stays identical across re-runs.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resolution", type=int, default=128, help="marching cubes grid")
    ap.add_argument("--fscore-tau", type=float, default=0.02)
    ap.add_argument("--no-mesh", action="store_true", help="skip chamfer/f-score/NC (much faster)")
    ap.add_argument("--sample-posterior", action="store_true",
                    help="decode z ~ q(z|x) instead of z = mu. Matches what the decoder actually saw "
                         "during training; z=mu hands it a latent whose scale is ~2x smaller than "
                         "anything it trained on (measured: ~9 V-IoU lower). Deterministic given "
                         "--seed, but it is one draw per shape, not the posterior mean.")
    ap.add_argument("--mc-chunk", type=int, default=16384,
                    help="query points per decoder call when marching-cubing. The default 65536 "
                         "makes one kernel long enough to hit the display driver's watchdog on a "
                         "GPU that is also driving a desktop (cudaErrorLaunchTimeout); smaller "
                         "chunks cost a little throughput and remove that failure mode.")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partial", default="results/paper_eval_partial.jsonl",
                    help="resume sidecar; delete it to force a clean recompute")
    ap.add_argument("--md-out", default="docs/eval_result.md")
    ap.add_argument("--json-out", default="results/paper_eval.json")
    args = ap.parse_args()

    if not args.no_mesh and args.fscore_tau <= 2 / args.resolution:
        raise SystemExit(f"--fscore-tau {args.fscore_tau} must exceed the MC cell size "
                         f"2/{args.resolution}={2/args.resolution:.4f}, else it scores "
                         f"discretisation noise rather than the model")

    torch.set_num_threads(args.threads)
    runs = [c.split("=", 1) for c in args.ckpt]
    for _, p in runs:
        if not os.path.isfile(p):
            raise SystemExit(f"checkpoint not found: {p}")

    # Refuse to start rather than OOM hours in. Measured high-water RESERVED bytes (the figure that
    # actually competes with other processes) on this model at sample_posterior, mc_chunk 16384:
    # res 128 -> 2.6 GiB, res 256 -> 7.5 GiB, of which the cached grid Fourier embedding alone is
    # resolution^3 * 51 * 4 bytes (0.40 GiB at 128, 3.19 GiB at 256). A concurrent VAE training run
    # holds ~14.7 GiB, so 256 and training cannot share this card at all.
    if args.device.startswith("cuda") and not args.no_mesh:
        need = {128: 2.6, 256: 7.5}.get(args.resolution,
                                        2.6 * (args.resolution / 128) ** 3) + 1.5   # +margin
        free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
        print(f"vram: {free:.2f} GiB free of {total:.2f} GiB; "
              f"resolution {args.resolution} needs about {need:.1f} GiB")
        if free < need:
            raise SystemExit(
                f"only {free:.2f} GiB free but resolution {args.resolution} needs ~{need:.1f} GiB. "
                f"Stop whatever else is on the GPU (a VAE training run holds ~14.7 GiB), or pass "
                f"--resolution 128, or --no-mesh to skip the marching-cubes metrics entirely.")

    # One fixed shape set for every checkpoint -- the whole point of this script.
    mesh_paths = sorted(glob.glob(os.path.join("data", args.split, "**", "*.off.gz"), recursive=True))
    if not mesh_paths:
        raise SystemExit(f"no meshes under data/{args.split}")
    if args.shapes_file:
        rels = [l.strip() for l in open(args.shapes_file) if l.strip()]
        known = {os.path.splitext(os.path.relpath(m, os.path.join("data", args.split))
                                  .removesuffix(".gz"))[0] for m in mesh_paths}
        missing = [r for r in rels if r not in known]
        if missing:
            raise SystemExit(f"{len(missing)} ids in {args.shapes_file} are not in data/{args.split}, "
                             f"first: {missing[0]}")
        n = len(rels)
        print(f"shape list: {n} ids from {args.shapes_file}")
    else:
        n = len(mesh_paths) if args.shapes <= 0 else min(args.shapes, len(mesh_paths))
        pick = sorted(np.random.default_rng(args.seed).choice(len(mesh_paths), n, replace=False))
        rels = [os.path.splitext(os.path.relpath(mesh_paths[i], os.path.join("data", args.split))
                                 .removesuffix(".gz"))[0] for i in pick]
    work = [(r, os.path.join("cache", r.split(os.sep)[0], "4_watertight_scaled",
                             os.path.basename(r) + ".pt")) for r in rels]
    work = [(r, c) for r, c in work if os.path.isfile(c)]
    print(f"{len(work)}/{n} shapes from data/{args.split} have a cache; evaluating those")

    # --- resume ---
    os.makedirs(os.path.dirname(args.partial) or ".", exist_ok=True)
    fp = fingerprint(args)
    done, stale = {}, 0
    if os.path.isfile(args.partial):
        with open(args.partial) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a half-written last line from a hard kill
                if r.get("fp") != fp:
                    stale += 1
                    continue
                done[(r["ckpt"], r["rel"])] = r
        print(f"resume: {len(done)} usable shape-results in {args.partial}")
        if stale:
            print(f"        {stale} records ignored -- different eval settings ({fp})")
    print(f"eval mode: z = {'sample from q(z|x)' if args.sample_posterior else 'mu (posterior mode)'}")

    results, order, skipped = {}, [c[0] for c in runs], []
    for name, ckpt in runs:
        t0 = time.time()
        todo = [(r, c) for r, c in work if (name, r) not in done]
        model = pre = post = smp = None
        epoch = None
        if todo:
            model, pre, cfg, epoch = load_run(ckpt, args.device)
            smp = sample_fn(pre, cfg)
            post = Postprocess(pre.fourier_embedder, resolution=args.resolution, chunk=args.mc_chunk)
        else:  # everything cached, but the table still needs the epoch
            sd = torch.load(ckpt, map_location="meta", weights_only=False)
            epoch = sd.get("epoch")

        with open(args.partial, "a") as sidecar:
            bar = tqdm(todo, desc=f"{name}", unit="shape", dynamic_ncols=True)
            for rel, cpath in bar:
                try:
                    cache = torch.load(cpath, weights_only=False)
                    rec = eval_shape(model, pre, post, smp, cache,
                                     os.path.join("data", args.split, rel + ".off.gz"), args)
                except Exception as e:
                    # A CUDA failure poisons the context: every later call raises too, so
                    # "skip and continue" would silently burn the rest of the sweep producing
                    # nothing (measured: 2177 of 2592 shapes skipped in 31 min after one
                    # cudaErrorLaunchTimeout). Only genuinely per-shape faults are skippable.
                    if isinstance(e, torch.cuda.CudaError) or "CUDA" in str(e) or \
                            type(e).__name__ == "AcceleratorError":
                        bar.close()
                        raise RuntimeError(
                            f"CUDA failure on {rel} ({type(e).__name__}: {e}). The context cannot "
                            f"recover -- aborting. {len(done)} shape-results are already saved in "
                            f"{args.partial}; re-run the same command to continue from there."
                        ) from e
                    bar.write(f"  [{name}] {rel}: {type(e).__name__}: {e} -- skipped")
                    skipped.append((name, rel, f"{type(e).__name__}: {e}"))
                    continue
                rec.update(ckpt=name, rel=rel, fp=fp)
                sidecar.write(json.dumps(rec) + "\n")
                sidecar.flush()
                os.fsync(sidecar.fileno())     # survive a hard kill, not just a clean exit
                done[(name, rel)] = rec

        recs = [done[(name, r)] for r, _ in work if (name, r) in done]
        results[name] = aggregate(recs) | {"ckpt": ckpt, "epoch": epoch,
                                           "secs": round(time.time() - t0, 1)}
        del model
        if args.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass          # never let cleanup replace the error that actually mattered

    txt, md = render(results, order, args)
    print("\n" + txt)
    if skipped:
        print(f"\n{len(skipped)} shape(s) skipped on a per-shape fault:")
        for name, rel, msg in skipped:
            print(f"  [{name}] {rel}: {msg}")

    os.makedirs(os.path.dirname(args.md_out) or ".", exist_ok=True)
    with open(args.md_out, "w") as f:      # created if absent, overwritten if present
        f.write(md)
    print(f"\nmarkdown -> {args.md_out}")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"json     -> {args.json_out}")
    print(f"partial  -> {args.partial}  (delete to force a clean recompute)")


if __name__ == "__main__":
    main()

"""
    uv run python -m scripts.paper_eval \
        --ckpt lambda=outputs/.../best.pt --ckpt disjoint=outputs/.../best.pt \
        --shapes 0 --device cuda --md-out docs/eval_result.md
"""