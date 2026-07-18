#!/usr/bin/env python
"""Smoke 04 — microphone. Records 2s, saves a wav, plays it back.

macOS grants Microphone access to the terminal app, not to Python. When it is
denied, sounddevice usually does NOT raise: it hands back a buffer of digital
silence. A test that only checked "did it record" would pass on a dead mic, so
this checks the signal level and treats near-silence as a failure.
"""

from __future__ import annotations

import numpy as np
from _bootstrap import banner, fail, ok, parse_args, permission_hint, skip

from armani import config
from armani.logutil import log_event

# Peak amplitude below this on a 2s take means nothing reached the ADC.
SILENCE_PEAK = 1e-4


def main() -> int:
    args = parse_args(__doc__ or "")
    banner("Smoke 04: microphone")

    wav_path = config.TEST_OUT_DIR / "mic_test.wav"
    if args.dry_run:
        print(f"[dry-run] would record {config.MIC_TEST_SECONDS:.0f}s at "
              f"{config.MIC_SAMPLE_RATE} Hz, {config.MIC_CHANNELS}ch")
        print(f"[dry-run] would save {wav_path} and play it back")
        return ok("dry run complete")

    try:
        import sounddevice as sd
        import soundfile as sf
    except (ImportError, OSError) as exc:
        return fail(f"audio libraries unavailable: {type(exc).__name__}: {exc}")

    try:
        default_in = sd.query_devices(kind="input")
        default_out = sd.query_devices(kind="output")
    except Exception as exc:
        return fail(f"no audio devices: {type(exc).__name__}: {exc}")

    print(f"Input : {default_in['name']}")
    print(f"Output: {default_out['name']}")
    print(
        "\nREMINDER: the demo mic is the wired headset, not the C920's mic.\n"
        "Set it as the macOS input device before the demo."
    )

    frames = int(config.MIC_TEST_SECONDS * config.MIC_SAMPLE_RATE)
    print(f"\nRecording {config.MIC_TEST_SECONDS:.0f}s — say something now...")
    try:
        recording = sd.rec(
            frames,
            samplerate=config.MIC_SAMPLE_RATE,
            channels=config.MIC_CHANNELS,
            dtype="float32",
        )
        sd.wait()
    except Exception as exc:
        permission_hint("Microphone", f"recording failed: {exc}")
        return fail(f"could not record: {type(exc).__name__}: {exc}")

    peak = float(np.max(np.abs(recording))) if recording.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(recording)))) if recording.size else 0.0
    print(f"Captured {recording.shape[0]} frames | peak {peak:.4f} | rms {rms:.4f}")

    config.TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(wav_path), recording, config.MIC_SAMPLE_RATE)
    except Exception as exc:
        return fail(f"could not write {wav_path}: {exc}")
    print(f"Saved {wav_path}")

    if peak < SILENCE_PEAK:
        permission_hint(
            "Microphone",
            f"recorded digital silence (peak {peak:.6f}) — the mic is muted or access is denied",
        )
        return fail("microphone produced silence")

    print("\nPlaying it back...")
    try:
        sd.play(recording, config.MIC_SAMPLE_RATE)
        sd.wait()
    except Exception as exc:
        log_event("smoke_04", peak=peak, rms=rms, playback_error=str(exc))
        return skip(f"recording worked (peak {peak:.4f}) but playback failed: {exc}")

    log_event("smoke_04", peak=peak, rms=rms, path=str(wav_path))
    return ok(f"recorded and played back {config.MIC_TEST_SECONDS:.0f}s (peak {peak:.4f})")


if __name__ == "__main__":
    raise SystemExit(main())
