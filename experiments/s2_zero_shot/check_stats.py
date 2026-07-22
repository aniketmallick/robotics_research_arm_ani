"""Q1 measurement-integrity check for Spike S2 (headless, no arm, no operator).

Answers the ONE question that decides whether S2's "near-total failure" is a real
zero-shot result or a harness bug: **did the postprocessor actually load the
checkpoint's action unnormalize stats?** For MEAN_STD, the unnormalize step raises
if mean/std are missing, so a clean run already implies they loaded — but this
prints the actual per-joint mean/std so the claim is verified, not inferred, and
so we can read WHICH convention the base model's actions speak.

It then runs inference on the synthetic gray frame AND a REAL camera frame
(default logs/last_frame.jpg — an actual table image), printing both the
normalized action (pre-unnormalize) and the final action. If the normalized
values sit near 0, the model is regressing to its training mean on our OOD input;
the final magnitude then just reflects that mean's scale.

    python -m experiments.s2_zero_shot.check_stats
    python -m experiments.s2_zero_shot.check_stats --frame tests/out/detect.jpg --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.s2_zero_shot import camera, smolvla_io

DEFAULT_FRAME = "logs/last_frame.jpg"
TASK = "Pick up the red block"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _fmt(arr) -> list:
    return np.round(np.asarray(arr, dtype=float).ravel(), 3).tolist()


def _run_once(policy, pre, post, spec, bgr):
    """Return (normalized_action, final_action) for one frame."""
    import torch

    frame = smolvla_io.build_frame({j: 0.0 for j in smolvla_io.JOINT_ORDER}, bgr, TASK, spec)
    policy.reset()
    with torch.no_grad():
        batch = pre(frame)
        normalized = policy.select_action(batch)      # pre-unnormalize (model space)
        final = post(normalized)                       # unnormalized
    norm = normalized.detach().to("cpu").numpy().ravel()
    fin = final.detach().to("cpu").numpy().ravel() if hasattr(final, "detach") else np.asarray(final).ravel()
    return norm, fin


def main(argv: list[str] | None = None) -> int:
    import cv2
    import torch

    parser = argparse.ArgumentParser(prog="check_stats", description="S2 unnormalize-stats + real-frame check.")
    parser.add_argument("--frame", default=DEFAULT_FRAME, help="real image to test (default logs/last_frame.jpg)")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--samples", type=int, default=3, help="inferences per frame (flow-matching is stochastic)")
    args = parser.parse_args(argv)

    print(f"loading lerobot/smolvla_base on {args.device} ...")
    policy, pre, post, spec = smolvla_io.load(args.device)
    print(spec.summary())

    # --- Q1(a): are the action unnormalize stats loaded AND routed? ---
    step = smolvla_io.stats_step(post)
    print(f"\n=== Q1(a) postprocessor unnormalize stats (routed dataset: {spec.stats_dataset}) ===")
    print("routed features:", list(spec.routed_features))
    if step is None:
        print("!! could not locate a stats-bearing step in the postprocessor — INVESTIGATE")
    else:
        print("norm_map:", getattr(step, "norm_map", None))
        action_stats = (getattr(step, "stats", {}) or {}).get("action")
        if not action_stats or "mean" not in action_stats or "std" not in action_stats:
            print("!! ACTION mean/std ABSENT after routing -> unnormalize is a no-op. The outputs "
                  "are raw normalized values (HARNESS BUG), not a zero-shot result. Check the "
                  "dataset name against the checkpoint's '<ds>.buffer.action' keys.")
        else:
            mean, std = np.asarray(action_stats["mean"], float), np.asarray(action_stats["std"], float)
            print("action mean:", _fmt(mean))
            print("action std :", _fmt(std))
            identity = np.allclose(mean, 0.0) and np.allclose(std, 1.0)
            print("VERDICT:", "IDENTITY (mean 0 / std 1) -> stats effectively absent" if identity
                  else "REAL stats routed -> unnormalization is genuine (base arm's convention)")

    # --- input side: did STATE normalization stats load too? ---
    pre_step = smolvla_io.stats_step(pre)
    print("\n=== preprocessor (input) state-normalize stats ===")
    if pre_step is None:
        print("!! no stats step in preprocessor")
    else:
        state_stats = (getattr(pre_step, "stats", {}) or {}).get("observation.state")
        if state_stats and "mean" in state_stats:
            print("observation.state mean:", _fmt(state_stats["mean"]), "std:", _fmt(state_stats["std"]))
            print("STATE VERDICT: normalized")
        else:
            print("STATE VERDICT: absent -> state is fed RAW (this checkpoint's shared stats file "
                  "carries action stats only). Does NOT change the regress-to-mean result: the "
                  "model ignores the scene regardless, and only the ACTION path reaches the motors.")

    # --- Q1(b): synthetic vs a REAL frame ---
    syn_step0 = camera.synthetic_frame(0)
    frame_path = (REPO_ROOT / args.frame) if not Path(args.frame).is_absolute() else Path(args.frame)
    real = cv2.imread(str(frame_path))
    print(f"\n=== Q1(b) inference: synthetic vs real frame ({frame_path}) ===")
    print("real frame:", "MISSING — pass --frame <path>" if real is None else f"{real.shape}")

    sources = [("synthetic", syn_step0)] + ([("real", real)] if real is not None else [])
    for name, img in sources:
        print(f"\n[{name}]")
        for i in range(args.samples):
            norm, fin = _run_once(policy, pre, post, spec, img)
            print(f"  sample {i}: normalized={_fmt(norm)}  ->  final={_fmt(fin)}  |final|max={np.abs(fin).max():.3f}")

    print("\nReading guide: normalized≈0 means the model regresses to its training mean on our "
          "OOD input; final = normalized*std + mean, so final's scale is set by the stats above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
