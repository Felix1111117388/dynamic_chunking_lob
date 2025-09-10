# -*- coding: utf-8 -*-
# train_dc_lite.py  (Early Stopping + Resume + Periodic Checkpoints + Plots)

import os, sys, argparse, math, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- make local imports work when run from anywhere ---
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if FILE_DIR not in sys.path:
    sys.path.insert(0, FILE_DIR)

from dc_lite import DCLiteLM

# ---------------- dataset ----------------
class ByteSequenceDataset(Dataset):
    def __init__(self, paths, seq_len=2048, step=None, bytes_cap=0):
        blobs = []
        for p in paths:
            with open(p, "rb") as f:
                blobs.append(f.read())
        raw = b"".join(blobs)
        if bytes_cap and bytes_cap > 0:
            raw = raw[:bytes_cap]  # cap BEFORE building array

        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        self.data = torch.from_numpy(arr)
        self.seq_len = int(seq_len)
        self.step = int(step) if step else self.seq_len

        if len(self.data) < (self.seq_len + 1):
            raise ValueError(
                f"Dataset too small for seq_len={self.seq_len}: {len(self.data)} bytes available."
            )

        # compute starts on final (possibly capped) length
        self.starts = torch.arange(0, len(self.data) - self.seq_len - 1, self.step)

    def __len__(self): 
        return self.starts.numel()

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        x = self.data[s:s+self.seq_len].clone()
        y = self.data[s+1:s+self.seq_len+1].clone()
        return x, y

def device_auto():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def evaluate(model, loader, device, autocast_dtype=None,
             collect_chunks=False, max_hist=0):
    """
    Returns: (avg_loss, ppl, bpb, chunk_len_samples)
    If collect_chunks is True and the model supports return_chunks=True,
    this gathers up to max_hist chunk-length samples from VAL.
    """
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    tot, ntok = 0.0, 0
    chunk_lens = []
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device)
            if autocast_dtype:
                with torch.autocast(device.type, dtype=autocast_dtype):
                    logits, aux_loss, stats = model(
                        x, return_aux=True, return_chunks=collect_chunks)
                    loss = crit(logits.view(-1,256), y.view(-1)) + aux_loss
            else:
                logits, aux_loss, stats = model(
                    x, return_aux=True, return_chunks=collect_chunks)
                loss = crit(logits.view(-1,256), y.view(-1)) + aux_loss

            tot += loss.item()
            ntok += y.numel()

            if collect_chunks and isinstance(stats, dict) and "chunk_len_samples" in stats and max_hist > 0:
                need = max_hist - len(chunk_lens)
                if need > 0:
                    sl = stats["chunk_len_samples"]
                    if len(sl) > need:
                        sl = sl[:need]
                    chunk_lens.extend(sl)

    avg = tot/ntok
    ppl = math.exp(avg)
    bpb = avg / math.log(2.0)
    return avg, ppl, bpb, chunk_lens

def plot_curves(history, outdir, title_prefix=""):
    os.makedirs(outdir, exist_ok=True)

    # Try SciencePlots style if available
    try:
        import scienceplots  # pip install SciencePlots
        plt.style.use("science")
    except Exception:
        pass

    ep = list(range(1, len(history.get("val_loss", [])) + 1))

    # Loss
    plt.figure()
    if "train_loss" in history: plt.plot(ep, history["train_loss"], label="train")
    if "val_loss"   in history: plt.plot(ep, history["val_loss"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("avg CE loss (nats/byte)")
    if title_prefix: plt.title(f"{title_prefix} Loss")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curves.png"), dpi=160)

    # Perplexity
    plt.figure()
    if "val_ppl" in history: plt.plot(ep, history["val_ppl"], label="val ppl")
    plt.xlabel("epoch"); plt.ylabel("perplexity")
    if title_prefix: plt.title(f"{title_prefix} Perplexity")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "perplexity.png"), dpi=160)

    # (1) Avg chunk length vs epoch
    if "avg_chunk_len" in history and len(history["avg_chunk_len"]) == len(ep):
        plt.figure()
        plt.plot(ep, history["avg_chunk_len"], label="avg chunk length")
        plt.xlabel("epoch"); plt.ylabel("avg chunk length (bytes)")
        if title_prefix: plt.title(f"{title_prefix} Avg Chunk Length")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "avg_chunk_len.png"), dpi=160)

    # (2) Histogram + ECDF of chunk lengths (VAL)
    if "chunk_len_samples" in history and len(history["chunk_len_samples"]) > 0:
        L = np.asarray(history["chunk_len_samples"], dtype=np.int64)

        # Histogram
        plt.figure()
        med = int(np.median(L))
        bins_max = max(32, min(med*2, int(L.max())))
        bins = min(100, bins_max)
        plt.hist(L, bins=bins)
        plt.xlabel("chunk length (bytes)"); plt.ylabel("count")
        if title_prefix: plt.title(f"{title_prefix} Chunk Length Histogram")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "chunk_len_hist.png"), dpi=160)

        # ECDF
        plt.figure()
        Ls = np.sort(L)
        y = np.arange(1, len(Ls)+1) / len(Ls)
        plt.plot(Ls, y)
        plt.xlabel("chunk length (bytes)"); plt.ylabel("ECDF")
        if title_prefix: plt.title(f"{title_prefix} Chunk Length ECDF")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "chunk_len_ecdf.png"), dpi=160)

    # (5) Learning rate vs epoch
    if "lr" in history and len(history["lr"]) == len(ep):
        plt.figure()
        plt.plot(ep, history["lr"])
        plt.xlabel("epoch"); plt.ylabel("learning rate")
        if title_prefix: plt.title(f"{title_prefix} Learning Rate")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "lr_curve.png"), dpi=160)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--val_files",   nargs="+", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--target_chunk_len", type=int, default=64)
    ap.add_argument("--aux_w", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--outdir", default="runs/dc_lite")
    ap.add_argument("--amp", action="store_true")
    # Early stopping
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="stop if no val improvement for N epochs (0 disables)")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="minimum decrease in val loss to count as an improvement")
    # Resume + checkpoints
    ap.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint (best.pt/last.pt/epoch_*.pt)")
    ap.add_argument("--save_every", type=int, default=0,
                    help="save epoch_{N}.pt every N epochs (0=off)")
    ap.add_argument("--save_last", action="store_true",
                    help="save last.pt each epoch")
    # Fast trials / loader tweaks
    ap.add_argument("--train_bytes_cap", type=int, default=0,
                    help="truncate train stream to first N bytes (0=all)")
    ap.add_argument("--val_bytes_cap", type=int, default=0,
                    help="truncate val stream to first N bytes (0=all)")
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
    # DC plots
    ap.add_argument("--collect_chunk_hist", action="store_true",
                    help="collect VAL chunk lengths for histogram/ECDF (requires model support)")
    ap.add_argument("--max_hist_samples", type=int, default=50000,
                    help="max chunk-length samples to collect on VAL")
    # NEW: skip plots for random search
    ap.add_argument("--no_plots", action="store_true",
                    help="skip generating/saving plots (useful for fast trials)")
    args = ap.parse_args()

    device = device_auto()
    print("Device:", device)

    train_ds = ByteSequenceDataset(
        args.train_files, seq_len=args.seq_len, step=args.seq_len, bytes_cap=args.train_bytes_cap
    )
    val_ds = ByteSequenceDataset(
        args.val_files,   seq_len=args.seq_len, step=args.seq_len, bytes_cap=args.val_bytes_cap
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type=="cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type=="cuda")
    )

    model = DCLiteLM(
        vocab_size=256,
        d_model_tok=256, d_model_chunk=384,
        n_layers_tok=2, n_heads_tok=4,
        n_layers_chunk=4, n_heads_chunk=6,
        mlp_mult=2.0, dropout=args.dropout,
        target_chunk_len=args.target_chunk_len,
        boundary_rate_weight=args.aux_w,
        smooth_tau=args.tau,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    try:
        # fused=True can be faster on A100; ignore if not supported
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, fused=True)
    except TypeError:
        pass
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(reduction="mean")

    autocast_dtype = None
    if args.amp:
        if device.type=="cuda" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
        if device.type=="cuda":
            torch.set_float32_matmul_precision("high")

    history={"train_loss":[], "val_loss":[], "val_ppl":[], "val_bpb":[],
             "lr":[], "avg_chunk_len":[]}
    best=float("inf")
    patience_ctr = 0
    start_epoch = 1
    os.makedirs(args.outdir, exist_ok=True)

    # --------- RESUME (optional) ----------
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:   opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            try: sched.load_state_dict(ckpt["sched"])
            except Exception: pass
        best = ckpt.get("best_val", best)
        if "history" in ckpt and isinstance(ckpt["history"], dict):
            for k,v in ckpt["history"].items():
                history[k] = v
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val={best:.4f})")
        if "sched" not in ckpt:
            for _ in range(max(0, start_epoch-1)): sched.step()
    # --------------------------------------

    try:
        for ep in range(start_epoch, args.epochs+1):
            model.train()
            run, ntok = 0.0, 0
            opt.zero_grad(set_to_none=True)
            stats = {"avg_chunk_len": float("nan")}  # last-batch DC stat for logging

            for it,(x,y) in enumerate(train_loader,1):
                x=x.to(device); y=y.to(device)
                if autocast_dtype:
                    with torch.autocast(device.type, dtype=autocast_dtype):
                        logits, aux_loss, stats = model(x, return_aux=True)
                        loss = crit(logits.view(-1,256), y.view(-1)) + aux_loss
                else:
                    logits, aux_loss, stats = model(x, return_aux=True)
                    loss = crit(logits.view(-1,256), y.view(-1)) + aux_loss

                (loss/args.accum).backward()
                if it % args.accum == 0:
                    if args.grad_clip: nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step(); opt.zero_grad(set_to_none=True)

                run += loss.item() * y.numel()
                ntok += y.numel()

            train_loss = run/ntok
            val_loss, val_ppl, val_bpb, chunk_lens = evaluate(
                model, val_loader, device, autocast_dtype,
                collect_chunks=args.collect_chunk_hist,
                max_hist=args.max_hist_samples
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)
            history["val_bpb"].append(val_bpb)
            history["lr"].append(float(opt.param_groups[0]["lr"]))
            history["avg_chunk_len"].append(float(stats.get("avg_chunk_len", float("nan"))))

            if args.collect_chunk_hist and chunk_lens:
                history["chunk_len_samples"] = history.get("chunk_len_samples", []) + list(chunk_lens)

            print(f"Epoch {ep:02d} | lr {opt.param_groups[0]['lr']:.6g} | "
                  f"train {train_loss:.4f} | val {val_loss:.4f} | ppl {val_ppl:.2f} | "
                  f"bpb {val_bpb:.3f} | avg_chunk_len~{history['avg_chunk_len'][-1]:.1f}")

            # ----- checkpoint payload -----
            ckpt_payload = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": ep,
                "best_val": best,
                "history": history,
                "config": vars(args),
            }

            # save-best + early stop
            improved = (best - val_loss) > args.early_stop_min_delta
            if improved:
                best = val_loss
                torch.save(ckpt_payload, os.path.join(args.outdir, "best.pt"))
                patience_ctr = 0
            else:
                patience_ctr += 1

            # optional periodic & last
            if args.save_last:
                torch.save(ckpt_payload, os.path.join(args.outdir, "last.pt"))
            if args.save_every and (ep % args.save_every == 0):
                torch.save(ckpt_payload, os.path.join(args.outdir, f"epoch_{ep}.pt"))

            if args.early_stop_patience and patience_ctr >= args.early_stop_patience:
                print(f"Early stopping at epoch {ep} (no val improvement for {patience_ctr} epochs).")
                break

            sched.step()

    except KeyboardInterrupt:
        print("Interrupted — saving last.pt")
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "epoch": ep, "best_val": best,
            "history": history, "config": vars(args),
        }, os.path.join(args.outdir, "last.pt"))
        raise

    with open(os.path.join(args.outdir,"history.json"),"w") as f:
        json.dump(history,f,indent=2)

    # ONLY plot if not disabled
    if not args.no_plots:
        plot_curves(history, args.outdir, title_prefix="DC-Lite")

    print("Saved artifacts to:", args.outdir)

if __name__=="__main__":
    main()
