"""Part A — headless benchmark of lerobot/smolvla_base on this Mac. No robot.

Loads the base checkpoint, prints/records its real feature spec, and times
inference on MPS and CPU with a synthetic observation. Doubles as our M1-class
VLA latency benchmark. Writes everything to ``env_report.md``.

Latency subtlety: SmolVLA is a *chunked* policy. ``select_action`` runs the model
once per chunk (chunk_size=50) and pops from a queue in between, so timing it
would mostly time a ``deque.popleft``. We time ``predict_action_chunk`` instead,
which always runs the model — the true cost of one VLA inference — and report the
amortized per-step cost (chunk latency / chunk_size) so the control-rate story is
honest.

    python -m experiments.s2_zero_shot.bench                 # both devices
    python -m experiments.s2_zero_shot.bench --devices mps   # just one
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from experiments.s2_zero_shot import camera, smolvla_io

_HERE = Path(__file__).resolve().parent
ENV_REPORT = _HERE / "env_report.md"
STEADY_CALLS = 20
TASK = "Pick up the red block"


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "n/a"


def collect_versions() -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lerobot": _pkg("lerobot"),
        "torch": torch.__version__,
        "transformers": _pkg("transformers"),
        "numpy": np.__version__,
        "opencv-python-headless": _pkg("opencv-python-headless"),
        "accelerate": _pkg("accelerate"),
        # Robot path deps (lazily require_package()'d by lerobot when motion.connect()
        # builds the SO-101 bus + processor pipeline). Not pulled by [smolvla].
        "feetech-servo-sdk": _pkg("feetech-servo-sdk"),
        "deepdiff": _pkg("deepdiff"),
        "pynput": _pkg("pynput"),
        "mps_available": str(torch.backends.mps.is_available()),
        "cuda_available": str(torch.cuda.is_available()),
    }


def _time_chunk_latency(policy, preprocessor, frame: dict, n: int) -> tuple[float, list[float]]:
    """Return (one_shot_s, steady_list_s) timing predict_action_chunk (model always runs)."""
    import torch

    def one_call() -> float:
        batch = preprocessor(frame)
        start = time.perf_counter()
        with torch.no_grad():
            policy.predict_action_chunk(batch)
        if policy.config.device == "mps":
            torch.mps.synchronize()  # MPS is async; sync so we time the real compute
        return time.perf_counter() - start

    one_shot = one_call()  # first call: includes lazy kernel compilation
    steady = [one_call() for _ in range(n)]
    return one_shot, steady


def bench_device(device: str, policy, spec) -> dict:
    import torch
    from lerobot.policies.factory import make_pre_post_processors

    policy.config.device = device
    move_start = time.perf_counter()
    policy.to(device)
    move_s = time.perf_counter() - move_start

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=smolvla_io.MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    # These are freshly-built processors, so route the per-dataset stats here too
    # (otherwise unnormalization silently no-ops and sample_action is raw normalized
    # values — the exact bug check_stats.py guards against).
    smolvla_io.route_dataset_stats(preprocessor, spec.stats_dataset)
    smolvla_io.route_dataset_stats(postprocessor, spec.stats_dataset)

    frame = smolvla_io.build_frame({j: 0.0 for j in smolvla_io.JOINT_ORDER}, camera.synthetic_frame(0), TASK, spec)

    # One real (postprocessed) action, to see whether outputs look like plausible
    # degrees or like normalized noise — the spike's key qualitative signal.
    policy.reset()
    sample_action = smolvla_io.infer(policy, preprocessor, postprocessor, frame)

    policy.reset()
    one_shot, steady = _time_chunk_latency(policy, preprocessor, frame, STEADY_CALLS)
    # Amortize over the re-plan interval the action queue actually uses
    # (n_action_steps), not chunk_size. Equal for smolvla_base (50 == 50), but a
    # variant with n_action_steps < chunk_size would replan sooner, so dividing by
    # chunk_size would understate the true per-step cost.
    replan_interval = getattr(policy.config, "n_action_steps", None) or spec.chunk_size
    return {
        "device": device,
        "move_to_device_s": move_s,
        "chunk_one_shot_s": one_shot,
        "chunk_steady_mean_s": statistics.mean(steady),
        "chunk_steady_median_s": statistics.median(steady),
        "chunk_steady_p90_s": sorted(steady)[int(0.9 * (len(steady) - 1))],
        "amortized_per_step_s": statistics.mean(steady) / replan_interval,
        "sample_action": np.round(sample_action, 3).tolist(),
        "sample_action_absmax": float(np.abs(sample_action).max()),
    }


def run(devices: list[str]) -> dict:
    import torch

    versions = collect_versions()
    print("versions:", versions)

    load_start = time.perf_counter()
    policy, _pre, _post, spec = smolvla_io.load(devices[0])
    load_s = time.perf_counter() - load_start
    print(f"loaded in {load_s:.1f}s | {spec.summary()}")

    results = []
    for device in devices:
        if device == "mps" and not torch.backends.mps.is_available():
            print(f"skip {device}: not available")
            continue
        print(f"benchmarking {device} ...")
        results.append(bench_device(device, policy, spec))
        for r in results[-1:]:
            print(
                f"  {device}: one-shot {r['chunk_one_shot_s'] * 1000:.0f} ms, "
                f"steady {r['chunk_steady_mean_s'] * 1000:.0f} ms/chunk "
                f"({r['amortized_per_step_s'] * 1000:.1f} ms/step amortized), "
                f"sample action absmax {r['sample_action_absmax']:.2f}"
            )

    report = {"versions": versions, "load_s": load_s, "spec": spec, "results": results}
    _write_env_report(report)
    print(f"wrote {ENV_REPORT}")
    return report


def _write_env_report(report: dict) -> None:
    spec = report["spec"]
    lines = ["# Spike S2 — env report (generated by bench.py)\n"]
    lines.append("## Parallel env `lerobot-vla` (ratified exception; the demo `lerobot` env is untouched)\n")
    lines.append("| package | version |")
    lines.append("|---|---|")
    for key, value in report["versions"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("Note: PyPI has no `lerobot==0.5.2`, so it was installed from the SAME local")
    lines.append("0.5.2 source tree the demo env uses — guaranteeing identical calibration/")
    lines.append("dataset formats. `python-dotenv` was added to this env so the runner can reuse")
    lines.append("the real `armani.safety.clamp_action`; the demo env got zero new packages.\n")
    lines.append("## Policy feature spec (discovered from lerobot/smolvla_base)\n")
    lines.append(f"- state_dim: **{spec.state_dim}** (our arm has 6 joints)")
    lines.append(f"- action_dim: **{spec.action_dim}** (mapped positionally onto our 6 joints — OOD)")
    lines.append(f"- cameras declared: **{list(spec.image_keys)}**; filled from our one C920: "
                 f"**{list(spec.fill_image_keys)}** (base: all three; a fine-tuned 1-camera checkpoint: just the one)")
    lines.append(f"- chunk_size / n_action_steps: **{spec.chunk_size}**")
    lines.append(f"- model construct + load time: **{report['load_s']:.1f} s**\n")
    lines.append("## Latency (synthetic observation)\n")
    lines.append("| device | move to dev (s) | one-shot chunk (ms) | steady mean (ms) | steady p90 (ms) | amortized /step (ms) | sample action absmax |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report["results"]:
        lines.append(
            f"| {r['device']} | {r['move_to_device_s']:.1f} | {r['chunk_one_shot_s'] * 1000:.0f} | "
            f"{r['chunk_steady_mean_s'] * 1000:.0f} | {r['chunk_steady_p90_s'] * 1000:.0f} | "
            f"{r['amortized_per_step_s'] * 1000:.1f} | {r['sample_action_absmax']:.2f} |"
        )
    lines.append("")
    lines.append("`sample action` (postprocessed, one call) per device — the honest OOD signal")
    lines.append("(plausible degrees vs normalized noise):\n")
    for r in report["results"]:
        lines.append(f"- **{r['device']}**: {r['sample_action']}")
    lines.append("")
    ENV_REPORT.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="Part A headless benchmark of smolvla_base.")
    parser.add_argument("--devices", nargs="+", default=["mps", "cpu"], choices=["mps", "cpu"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.devices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
