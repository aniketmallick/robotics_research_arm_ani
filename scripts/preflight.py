#!/usr/bin/env python
"""Go / no-go before you present. About 90 seconds, one table, one verdict.

    python scripts/preflight.py            # the full check
    python scripts/preflight.py --no-api   # skip the paid Gemini call

Run this after setting up the table and BEFORE the audience arrives. It checks
the things that have actually broken during this build, in the order they would
ruin the demo:

  * Gemini billing live — the free tier is 20 requests/day/model and one gated
    pick spends three or four. This makes a real (fractions of a cent) call and
    goes RED on 429, because a quota failure means G1 cannot run and therefore
    NO pick can even start.
  * zones, pick macros, gestures, verified home — the recorded assets.
  * camera, mic, kill-switch permissions — the physical path.
  * a dataset backup exists — the recordings live outside the repo.

FAIL means do not present until it is fixed. WARN means it will probably work
but you should know. Nothing here moves the arm.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: E402,F401  (verifies the interpreter, exits if wrong)

from armani import config, gestures, motion, zones  # noqa: E402
from armani.logutil import log_event  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
COLOUR = {PASS: GREEN, WARN: YELLOW, FAIL: RED}


@dataclass
class Result:
    status: str
    detail: str


def check_keys() -> Result:
    missing = [name for name in config.API_KEY_VARS if config.api_key(name) is None]
    if missing:
        return Result(FAIL, f"missing: {', '.join(missing)} — see .env.example")
    return Result(PASS, f"all {len(config.API_KEY_VARS)} present")


def check_gemini_billing(skip: bool) -> Result:
    """THE check. A real call, because only a real call proves the quota."""
    if skip:
        return Result(WARN, "skipped (--no-api) — you have NOT proven the quota")
    if config.api_key("GOOGLE_API_KEY") is None:
        return Result(FAIL, "GOOGLE_API_KEY is not set")

    from armani import eyes

    frame_path = config.TEST_OUT_DIR / "frame.jpg"
    try:
        import cv2

        frame = cv2.imread(str(frame_path)) if frame_path.is_file() else None
        if frame is None:
            frame = eyes.capture_frame()
    except Exception as exc:
        return Result(WARN, f"could not get a frame to test with: {exc}")

    started = time.perf_counter()
    try:
        # One sample, not two: this is a liveness probe, not a detection.
        detection = eyes.locate("anything", frame=frame, samples=1)
    except eyes.EyesError as exc:
        message = str(exc)
        if "quota" in message.lower() or "429" in message:
            return Result(FAIL, "QUOTA EXHAUSTED — enable billing in Google AI Studio. No pick can start.")
        return Result(FAIL, f"Gemini unusable: {message[:120]}")
    except Exception as exc:
        return Result(FAIL, f"Gemini call blew up: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - started
    answered = detection.model if detection is not None else ""
    primary = config.GEMINI_MODELS[0]
    if answered and answered != primary:
        # Something answered, so the demo works — but the primary being spent
        # means we are one model from total failure and paying its timeout on
        # every call. The operator should know BEFORE the audience arrives.
        return Result(WARN, f"answered by FALLBACK {answered} in {elapsed:.1f}s — {primary} is "
                            "likely quota-exhausted. Enable billing.")
    return Result(PASS, f"live ({answered or 'no detection, but no error'}) in {elapsed:.1f}s")


def check_zones() -> Result:
    zone_set = zones.load_zones()
    if zone_set is None:
        return Result(FAIL, "no zones — run: python scripts/define_zones.py")
    return Result(PASS, f"{len(zone_set)} spots: {', '.join(z.label for z in zone_set.zones)}")


def check_pick_macros() -> Result:
    recorded = gestures.episode_count(config.PICK_DATASET_ROOT)
    zone_set = zones.load_zones()
    needed = len(zone_set) if zone_set else 0
    if recorded == 0:
        return Result(FAIL, f"no pick macros at {config.PICK_DATASET_ROOT} — see docs/recording_picks.md")
    if needed and recorded < needed:
        return Result(FAIL, f"{recorded} macros for {needed} zones — the last {needed - recorded} spot(s) cannot be picked")
    return Result(PASS, f"{recorded} macros for {needed} zones")


def check_gestures() -> Result:
    recorded = gestures.episode_count()
    expected = len(config.GESTURES)
    if recorded == 0:
        return Result(FAIL, "no gestures recorded — see docs/recording_gestures.md")
    if recorded < expected:
        missing = [g for g in gestures.list_gestures() if config.GESTURES[g] >= recorded]
        return Result(WARN, f"{recorded}/{expected} — missing: {', '.join(missing)}")
    return Result(PASS, f"{recorded} gestures")


def check_home() -> Result:
    if not config.HOME_VERIFIED:
        return Result(FAIL, "home not verified — run: python scripts/capture_home.py (safety rule 4)")
    return Result(PASS, "verified")


def check_camera() -> Result:
    from armani import eyes

    if config.CAMERA_INDEX is None:
        return Result(WARN, "ARMANI_CAMERA_INDEX not set — smoke_03 finds it")
    try:
        frame = eyes.capture_frame()
    except eyes.EyesError as exc:
        return Result(FAIL, str(exc)[:140])
    height, width = frame.shape[:2]
    if (width, height) != (config.CAMERA_WIDTH, config.CAMERA_HEIGHT):
        return Result(FAIL, f"delivered {width}x{height}, but the zones were mapped at "
                            f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    return Result(PASS, f"index {config.CAMERA_INDEX} at {width}x{height}")


def check_zone_frame_match() -> Result:
    """The zones only mean anything at the size they were clicked at."""
    zone_set = zones.load_zones()
    if zone_set is None:
        return Result(WARN, "no zones to check")
    if zone_set.frame_size != (config.CAMERA_WIDTH, config.CAMERA_HEIGHT):
        return Result(FAIL, f"zones mapped at {zone_set.frame_size}, camera configured for "
                            f"{(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)} — re-run define_zones.py")
    return Result(PASS, f"zones and camera agree on {zone_set.frame_size[0]}x{zone_set.frame_size[1]}")


def check_microphone() -> Result:
    try:
        import sounddevice as sd

        name = str(sd.query_devices(kind="input")["name"])
    except Exception as exc:
        return Result(WARN, f"could not query the input device: {exc}")
    if any(bad.lower() in name.lower() for bad in config.AUDIO_DEVICE_WARN_SUBSTRINGS):
        return Result(WARN, f"default input is {name!r} — the demo mic is the WIRED HEADSET")
    return Result(PASS, name)


def check_input_monitoring() -> Result:
    """The global spacebar and the ESC kill switch both need this permission.

    ``listener.running`` is NOT the signal. On macOS an untrusted process starts
    a listener perfectly happily and then simply never receives an event, so
    checking only that it started reports the kill switch as fine while it is
    silently dead. pynput exposes the real answer as ``Listener.IS_TRUSTED``,
    taken from AXIsProcessTrusted().
    """
    try:
        from pynput import keyboard
    except Exception as exc:
        return Result(FAIL, f"pynput unavailable: {exc}")

    trusted = getattr(keyboard.Listener, "IS_TRUSTED", None)
    try:
        listener = keyboard.Listener(on_press=lambda key: None)
        listener.daemon = True
        listener.start()
        time.sleep(0.4)
        running = listener.running
        listener.stop()
    except Exception as exc:
        return Result(FAIL, f"listener would not start: {exc}")

    if not running:
        return Result(FAIL, "listener would not start — grant Accessibility + Input Monitoring to "
                            "your terminal (System Settings > Privacy & Security), then RESTART it")
    if trusted is False:
        return Result(FAIL,
                      "listener starts but this process is NOT TRUSTED, so it will never receive a "
                      "key: the ESC kill switch and push-to-talk are both dead. Grant Accessibility "
                      "+ Input Monitoring to THIS terminal app and restart it. "
                      "(Run preflight from the same terminal you will run the demo from.)")
    return Result(PASS, "trusted global key listener (ESC kill switch + push-to-talk)")


def check_serial() -> Result:
    ports = motion.find_serial_ports()
    if not ports:
        return Result(FAIL, f"no ports matching {config.SERIAL_PORT_GLOB} — is the arm plugged in?")
    if config.FOLLOWER_PORT and config.FOLLOWER_PORT not in ports:
        return Result(FAIL, f"ARMANI_FOLLOWER_PORT={config.FOLLOWER_PORT} is not present. Found: {', '.join(ports)}")
    return Result(PASS, config.FOLLOWER_PORT or f"{len(ports)} port(s): {', '.join(ports)}")


def check_decision_log() -> Result:
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with config.DECISION_LOG.open("a", encoding="utf-8"):
            pass
    except OSError as exc:
        return Result(FAIL, f"cannot write {config.DECISION_LOG}: {exc}")
    return Result(PASS, str(config.DECISION_LOG.relative_to(REPO_ROOT)))


def check_backup() -> Result:
    root = config.DATASET_BACKUP_DIR
    if not root.is_dir():
        return Result(WARN, "no dataset backup — run: python scripts/backup_datasets.py")
    backups = [p for label in root.iterdir() if label.is_dir() for p in label.iterdir() if p.is_dir()]
    if not backups:
        return Result(WARN, "backup directory is empty — run: python scripts/backup_datasets.py")
    newest = max(backups, key=lambda p: p.stat().st_mtime)
    age_h = (time.time() - newest.stat().st_mtime) / 3600
    if age_h > 48:
        return Result(WARN, f"newest backup is {age_h / 24:.1f} days old — re-run after any re-recording")
    return Result(PASS, f"{len(backups)} backup(s), newest {age_h:.1f}h old")


CHECKS: tuple[tuple[str, str], ...] = (
    ("API keys", "check_keys"),
    ("Gemini billing LIVE", "check_gemini_billing"),
    ("Zones defined", "check_zones"),
    ("Pick macros", "check_pick_macros"),
    ("Gestures", "check_gestures"),
    ("Home verified", "check_home"),
    ("Camera", "check_camera"),
    ("Zones match camera", "check_zone_frame_match"),
    ("Microphone", "check_microphone"),
    ("Kill switch permission", "check_input_monitoring"),
    ("Arm serial port", "check_serial"),
    ("Decision log writable", "check_decision_log"),
    ("Dataset backup", "check_backup"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-api", action="store_true", help="skip the real Gemini call")
    parser.add_argument("--dry-run", action="store_true", help="list the checks and exit")
    args = parser.parse_args()

    print("=" * 72)
    print("  ARM-ANI PRE-FLIGHT")
    print("=" * 72)

    if args.dry_run:
        for title, _ in CHECKS:
            print(f"  would check: {title}")
        return 0

    results: list[tuple[str, Result]] = []
    for title, func_name in CHECKS:
        print(f"  {title:<24} ... ", end="", flush=True)
        func = globals()[func_name]
        try:
            result = func(args.no_api) if func_name == "check_gemini_billing" else func()
        except Exception as exc:
            result = Result(FAIL, f"check itself blew up: {type(exc).__name__}: {exc}")
        print(f"{COLOUR[result.status]}{result.status}{RESET}  {DIM}{result.detail}{RESET}")
        results.append((title, result))

    failures = [t for t, r in results if r.status == FAIL]
    warnings = [t for t, r in results if r.status == WARN]

    print("=" * 72)
    if failures:
        print(f"  {RED}NO-GO{RESET} — {len(failures)} blocking: {', '.join(failures)}")
        print("  Fix these before presenting. Read the detail column above.")
    elif warnings:
        print(f"  {YELLOW}GO, WITH {len(warnings)} WARNING(S){RESET}: {', '.join(warnings)}")
    else:
        print(f"  {GREEN}GO{RESET} — everything checks out.")
    print("=" * 72)
    if not failures:
        print("\n  Next: python scripts/run_dashboard.py   (projector)")
        print("        python scripts/run_agent.py       (the demo)")
        print("        docs/demo_runbook.md              (the script)\n")

    log_event(
        "preflight",
        go=not failures,
        failures=failures,
        warnings=warnings,
        results={t: r.status for t, r in results},
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
