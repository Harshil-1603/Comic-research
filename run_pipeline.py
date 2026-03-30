"""
Master Pipeline — Comic Emotion Classification (Fully Automated)
================================================================
Converts PDFs → panels → auto-annotates via OCR+sentiment → trains → evaluates.

Usage:
    python run_pipeline.py                              # full pipeline
    python run_pipeline.py --start-from auto_annotate  # skip data prep
    python run_pipeline.py --only train eval            # run only these steps
    python run_pipeline.py --pdf data/source/mine.pdf  # specify PDF
    python run_pipeline.py --force-annotate            # re-annotate everything

Steps (in order):
    convert       — PDF → raw JPEGs
    extract       — Raw images → panel crops
    auto_annotate — OCR + sentiment → annotations.csv  (fully automated)
    split         — Stratified train/val/test split
    train         — Train fusion model
    eval          — Evaluate on test split
    ablation      — Run 4 ablation configs
    infer         — Demo inference on a sample panel
"""

import os
import sys
import time
import argparse
import subprocess
import glob

# Suppress HuggingFace tokenizer parallelism fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── ANSI colours ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def banner(msg):
    width = 62
    print(f"\n{BOLD}{CYAN}{'─'*width}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*width}{RESET}")


def ok(msg):   print(f"  {GREEN}✔  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠  {msg}{RESET}")
def fail(msg): print(f"  {RED}✘  {msg}{RESET}")
def info(msg): print(f"  {DIM}   {msg}{RESET}")


# ── Argument parsing ──────────────────────────────────────────────────────────

ALL_STEPS = ["convert", "extract", "auto_annotate", "split",
             "train", "eval", "ablation", "infer"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fully automated Comic Emotion Classification pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--pdf", type=str, default=None,
                   help="Source PDF (default: first PDF in data/source/)")
    p.add_argument("--start-from", type=str, choices=ALL_STEPS, default=None,
                   metavar="STEP", help="Skip all steps before STEP")
    p.add_argument("--skip", type=str, nargs="+", choices=ALL_STEPS, default=[],
                   metavar="STEP", help="Steps to skip")
    p.add_argument("--only", type=str, nargs="+", choices=ALL_STEPS, default=None,
                   metavar="STEP", help="Run only these steps")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs (default: 10)")
    p.add_argument("--ablation-epochs", type=int, default=5,
                   help="Epochs per ablation config (default: 5)")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    p.add_argument("--force-annotate", action="store_true",
                   help="Re-annotate all panels even if already done")
    p.add_argument("--infer-image", type=str, default=None,
                   help="Panel image for demo inference (auto-picked if omitted)")
    return p.parse_args()


# ── Step runner ───────────────────────────────────────────────────────────────

def run(cmd, step_name, fatal=True):
    """Run a command, stream its output, return success bool."""
    info(f"$ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    if result.returncode == 0:
        ok(f"{step_name} completed in {elapsed:.1f}s")
        return True
    if fatal:
        fail(f"{step_name} FAILED (exit {result.returncode})")
    else:
        warn(f"{step_name} exited with code {result.returncode} — continuing anyway.")
    return False


# ── Individual steps ──────────────────────────────────────────────────────────

def step_convert(args):
    banner("Step 1 / 8 — PDF → Raw Images")

    pdf = args.pdf
    if pdf is None:
        pdfs = sorted(glob.glob("data/source/*.pdf"))
        if not pdfs:
            warn("No PDF found in data/source/. Skipping convert.")
            warn("Place a PDF there or use --pdf path/to/file.pdf")
            return True   # non-fatal — user may already have raw images
        pdf = pdfs[0]
        info(f"Auto-detected: {pdf}")

    if not os.path.exists(pdf):
        fail(f"PDF not found: {pdf}")
        return False

    cmd = [
        sys.executable, "-c",
        (f"import sys; sys.path.insert(0,'.'); "
         f"from data.scripts.pdf_to_images import pdf_to_images; "
         f"pdf_to_images('{pdf}')")
    ]
    return run(cmd, "convert")


def step_extract(_args):
    banner("Step 2 / 8 — Extract Panels")
    raws = glob.glob("data/raw/*.jpg")
    if not raws:
        warn("data/raw/ is empty — skipping extraction.")
        return True
    return run([sys.executable, "data/scripts/run_extraction.py"], "extract")


def step_auto_annotate(args):
    banner("Step 3 / 8 — Auto-Annotation  (OCR + Sentiment)")

    panels = [f for f in glob.glob("data/processed/*.jpg") if "_debug" not in f]
    if not panels:
        fail("No panels in data/processed/. Run the extract step first.")
        return False

    import pandas as pd
    csv = "data/annotations.csv"
    done = 0
    if os.path.exists(csv) and not args.force_annotate:
        try:
            done = len(pd.read_csv(csv))
        except Exception:
            done = 0

    total = len(panels)
    info(f"{done}/{total} panels already annotated.")

    if done >= total and not args.force_annotate:
        ok("All panels already annotated — skipping.")
        return True

    cmd = [sys.executable, "data/scripts/auto_annotate.py"]
    if args.force_annotate:
        cmd.append("--force")
    return run(cmd, "auto_annotate")


def step_split(_args):
    banner("Step 4 / 8 — Stratified Train/Val/Test Split")

    csv = "data/annotations.csv"
    if not os.path.exists(csv):
        fail(f"{csv} not found. Run auto_annotate first.")
        return False

    import pandas as pd
    df = pd.read_csv(csv)
    if len(df) == 0:
        fail("annotations.csv is empty. Re-run auto_annotate.")
        return False

    class_counts = df["emotion"].value_counts()
    too_few = class_counts[class_counts < 2]
    if len(too_few):
        warn(f"Classes with < 2 samples (stratify may fail): {list(too_few.index)}")

    info(f"{len(df)} annotations across {df['emotion'].nunique()} emotions.")
    return run([sys.executable, "utils/split_data.py"], "split")


def step_train(args):
    banner("Step 5 / 8 — Train Model")

    if not os.path.exists("data/annotations_split.csv"):
        warn("No split CSV — training on full dataset (no val metrics).")

    cmd = [sys.executable, "train.py", "--epochs", str(args.epochs)]
    if args.device:
        cmd += ["--device", args.device]
    return run(cmd, "train")


def step_eval(args):
    banner("Step 6 / 8 — Evaluate on Test Split")

    ckpts = sorted(glob.glob("checkpoints/*.pt"))
    if not ckpts:
        warn("No checkpoints found — skipping eval.")
        return True   # non-fatal: can't eval without a checkpoint

    latest = ckpts[-1]
    info(f"Using checkpoint: {latest}")
    cmd = [sys.executable, "eval.py", "--checkpoint", latest]
    if args.device:
        cmd += ["--device", args.device]

    # Non-fatal: eval failure should never abort training results
    success = run(cmd, "eval", fatal=False)
    if not success:
        warn("Eval reported errors but pipeline will continue.")
    return True   # always continue to ablation + infer


def step_ablation(args):
    banner("Step 7 / 8 — Ablation Study (4 configs)")

    cmd = [sys.executable, "experiments/run_ablations.py",
           "--epochs", str(args.ablation_epochs)]
    if args.device:
        cmd += ["--device", args.device]
    # Non-fatal: ablation failure should not abort inference step
    success = run(cmd, "ablation", fatal=False)
    if not success:
        warn("Ablation reported errors but pipeline will continue.")
    return True


def step_infer(args):
    banner("Step 8 / 8 — Demo Inference")

    ckpts = sorted(glob.glob("checkpoints/*.pt"))
    if not ckpts:
        fail("No checkpoints. Run training first.")
        return False

    img = args.infer_image
    if img is None:
        panels = sorted([
            f for f in glob.glob("data/processed/*.jpg") if "_debug" not in f
        ])
        if not panels:
            warn("No panels found — skipping inference demo.")
            return True
        img = panels[0]
        info(f"Auto-selected: {img}")

    if not os.path.exists(img):
        fail(f"Image not found: {img}")
        return False

    cmd = [sys.executable, "inference.py", img]
    if args.device:
        cmd += ["--device", args.device]
    return run(cmd, "infer")


# ── Main ──────────────────────────────────────────────────────────────────────

STEP_FNS = {
    "convert":       step_convert,
    "extract":       step_extract,
    "auto_annotate": step_auto_annotate,
    "split":         step_split,
    "train":         step_train,
    "eval":          step_eval,
    "ablation":      step_ablation,
    "infer":         step_infer,
}


def main():
    args = parse_args()

    # Determine active steps
    if args.only:
        steps = [s for s in ALL_STEPS if s in args.only]
    else:
        steps = list(ALL_STEPS)
        if args.start_from:
            steps = steps[ALL_STEPS.index(args.start_from):]
        steps = [s for s in steps if s not in args.skip]

    print(f"\n{BOLD}Comic Emotion Classification — Automated Pipeline{RESET}")
    print(f"{DIM}Steps : {' → '.join(steps)}{RESET}")
    print(f"{DIM}Epochs: {args.epochs}  |  Ablation: {args.ablation_epochs}{RESET}")

    results = {}
    t_start = time.time()

    for step in steps:
        success = STEP_FNS[step](args)
        results[step] = success
        if not success:
            fail(f"Pipeline aborted at step '{step}'.")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Pipeline Summary")
    for step in steps:
        if step in results:
            sym = f"{GREEN}PASS{RESET}" if results[step] else f"{RED}FAIL{RESET}"
        else:
            sym = f"{YELLOW}SKIP{RESET}"
        print(f"  {sym}  {step}")

    print(f"\n{DIM}Total time: {time.time() - t_start:.1f}s{RESET}\n")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
