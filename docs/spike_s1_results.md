# Spike S1 — ChArUco recalibration + hover revival

**Question:** can the modular pipeline (Gemini point → homography → IK) hover the
gripper within ~1.5 cm over a named object, once calibration is done with a
printed ChArUco board instead of free-form tip clicks?

**Why stage 4 failed:** ~68 px honest reprojection, because 6 hand-made tip
clicks were fitting an 8-DOF pixel→robot homography — click noise warped the map.

**The fix (this spike):** split the map into two well-conditioned pieces.
- **pixel → board-mm**: a homography fit from the board's *sub-pixel* ChArUco
  corners. Many exact points → tiny RMS.
- **board → robot**: a **rigid 2D** transform (rotation + translation, 3 DOF)
  fit from just **3 tip touches**. Click error averages out over 3 DOF instead
  of distorting 8. No ruler for the board's robot position — the arm measures
  itself via FK.
- Compose to one 3×3 pixel→robot matrix; it drops into the existing
  `homography.json` and `grasp.hover_over` unchanged.

## Numbers

Synthetic / rendered-board (verified by me, no hardware):

| quantity | value |
|---|---|
| ChArUco detect + fit on a clean rendered board | **RMS 0.004 px**, 24/24 corners |
| rigid fit recovering a known 30° + (0.25, −0.05) m transform (0.5 mm click noise) | angle 29.75°, t (0.2498, −0.0500), residuals 0.15–0.44 mm |
| compose pixel→robot, single-corner check | **0.01 mm** vs expected |
| pixel→robot→pixel round-trip (matrix sanity) | ~5e-13 px |
| hover dry-run chain for a synthetic pixel | pixel (320,240) → robot (0.300, 0.000) m → IK reachable, 6.8 mm IK error, 30° lean |

Real hardware (NEEDS THE OPERATOR — fill in):

| quantity | value | how |
|---|---|---|
| board homography RMS (printed board, C920) | **TODO px** | `calibrate_charuco.py --run`, step 1 (gate ≤10 px) |
| per-corner tip residuals | **TODO / TODO / TODO mm** | `--run`, step 3 (target <5 mm) |
| ruler end-to-end error at 2 board corners | **TODO cm** | `--check --live-corners` |
| hover hits within ~1.5 cm | **TODO / 5** | `hover_object.py --object X --live`, 5 objects |
| parallax residual on the tallest object | **TODO cm** | note per object; expect 1–2 cm regardless |

## Operator runbook

1. `python scripts/make_charuco.py` → prints `armani/data/charuco_board.png`.
   Print at **100% / "Actual size"** (never "fit to page").
2. **Measure one square with a ruler**; `export ARMANI_CHARUCO_SQUARE_MM=<measured>`.
   The measured number is truth.
3. Tape the board FLAT on the table, fully in the C920 view. **Do not move the
   camera from here on.**
4. `python scripts/calibrate_charuco.py` (dry-run) → confirm **RMS ≤ 10 px** on
   screen. If higher: fix lighting/flatness/framing and retry. No tip-click
   fallback — that experiment already failed.
5. `python scripts/calibrate_charuco.py --run` → touch the **3 red-circled
   corners** with the gripper tip when prompted (torque is released; support the
   arm). Confirm per-point residuals look sane (<5 mm). It saves the map.
6. Optional: `python scripts/calibrate_charuco.py --check --live-corners` — hover
   over 2 corners, measure the offset with a ruler. That ruler number is the
   honest end-to-end error.
7. Remove the board. Place 5 objects at scattered positions. For each:
   `python scripts/hover_object.py --object "<name>" --live` (operator present,
   hand on ESC). Tip visually centred within ~1.5 cm on ≥4/5 = success.

## Boundary-margin + teardown fix (2026-07-23)

The first `--check --live-corners` run refused **both** corners. Root cause: the
corner targets are hull **vertices** of the saved polygon (the polygon is the
convex hull of the board corners themselves), and a strict point-in-polygon
rejects its own boundary; re-detection jitter can also nudge a corner epsilon
outside. **The saved calibration (0.1 px RMS) is fine and was NOT re-run.** Three
additive fixes:

- **Workspace margin:** `calibrate.point_in_polygon` gained `margin_m` (default
  0.0, behaviour unchanged); the hover/check paths pass `ARMANI_POLYGON_MARGIN_M`.
  **Default is 25 mm, not the 15 mm first guessed** — measured on the real 0.1 px
  calibration, the saved polygon is the board-corner hull *shrunk inward by
  `TABLE_MARGIN_M` (20 mm)*, so the corner targets sit **17.7–20.0 mm outside** it
  and 15 mm still refused the outermost corners (`--live-corners` picks exactly
  those). 25 mm undoes the 20 mm shrink + ~5 mm jitter headroom → **0/24 corners
  refused**, landing only ~5 mm past the measured hull (still solidly on the real
  table, which is far larger than the board), so safety rule 3 holds. Hard-capped
  at 50 mm. Fail-closed is unchanged: an empty/degenerate/collinear polygon or a
  non-finite coordinate (or non-finite margin) still refuses.
- **Exit teardown:** the scripts now stop the pynput ESC listener and release cv2
  handles before exit (`safety.release_kill_switch`), guarding the known macOS
  teardown segfault. If it still fires after work completes, it corrupts nothing.
- **Kill-switch honesty:** each `--live` path now prints a LOUD warning if the ESC
  listener is not trusted (Input Monitoring not granted) — ESC is dead, only Ctrl-C
  freezes. The trust check queries `AXIsProcessTrusted()` directly
  (`safety.esc_listener_trusted`); reading `Listener.IS_TRUSTED` off the *class* is
  a permanently-False default that cried wolf on every run (a HIGH the adversarial
  review caught), so `preflight.py` was fixed to share this one correct check.
  Warn, don't block.

> **TODO — next recalibration only; do NOT re-run anything now.** The two margins
> fight: the ChArUco path SHRINKS the hull by `TABLE_MARGIN_M` (20 mm) at build,
> then the check DILATES it back by 25 mm. That round-trip is a trap for a future
> reader. The clean endgame is `TABLE_MARGIN_M = 0` for the ChArUco path — the hull
> is already conservative (it is the board, not the table edge) — plus a ~10 mm
> check margin. Apply this ONLY at the next recalibration, never retroactively (the
> current saved 0.1 px map stays as-is).

## Parallax

The camera views the table at an angle, so a point on an object's *visual
centre* sits above the table plane and the homography maps it **behind** the
object. `eyes.locate(contact_point=True)` (used by `hover_object` unless
`--centre`) asks Gemini for the point where the object *meets the table*, which
removes most of that bias. Expect ~1–2 cm residual on tall objects regardless —
record it, don't fight it.

## Conclusion

**TODO after the print + arm run.** The math is sound and well-conditioned: on a
clean board the pixel→board fit is sub-pixel and the rigid composition is exact
to microns, so the remaining error budget is entirely (a) the printed-square
measurement, (b) the 3 tip touches, and (c) parallax. Whether that lands under
1.5 cm is the number the operator produces in steps 4–7. If board RMS comes back
in single digits and tip residuals under 5 mm, the ~1.5 cm target is plausible;
if RMS is >10 px the board or camera geometry is the problem, not the method —
stop and report per the time-box rule.
