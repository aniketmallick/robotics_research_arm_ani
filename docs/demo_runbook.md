# ARM-ANI demo runbook

Everything you need on the day, in the order you need it. Read this once the
night before and once at the venue.

**The thesis you are selling:** the arm can fail a grasp and you still win, as
long as the screen shows it *saw* the object, *stated* a number, *asked* when it
was unsure, and *stood down* when nobody answered. Trust is the product. The
grasp is a prop.

---

## T-30 minutes — setup

```bash
conda activate lerobot
cd ~/Documents/Claude/Projects/Interactive\ Robot
```

1. **Arm** — follower plugged in (USB *data* cable), powered, on the marked table.
2. **Camera** — C920 on its tripod. **Do not move it.** The zone pixels were
   clicked where it is now; a bump means re-running `define_zones.py`.
3. **Table** — the five marked spots visible, demo objects to hand:
   red block · marker pen · black cup · charger · airpod.
4. **Mic** — wired headset plugged in AND selected as the system input.
5. **Terminal permissions** — Accessibility *and* Input Monitoring granted to the
   terminal app you are about to use. Run preflight from **that same terminal**;
   the permission belongs to the app, not to Python.

```bash
python scripts/preflight.py
```

Green **GO** and nothing else will do. The two that bite:

| FAIL | fix |
|---|---|
| Gemini quota | enable billing in Google AI Studio (pay-as-you-go, fractions of a cent per call) |
| Kill switch permission | System Settings → Privacy & Security → Accessibility + Input Monitoring → your terminal → **restart the terminal** |

Then three windows:

```bash
python scripts/run_dashboard.py     # the PROOF screen:   http://localhost:8770
python scripts/run_avatar.py        # the FACE:           http://localhost:8771
python scripts/run_agent.py         # your terminal: the demo
```

**Screen layout that works:** dashboard on the projector (that is the one that
wins the argument), avatar on a second monitor, tablet or phone facing the
audience. For a phone, start it with `--host 0.0.0.0` and open the LAN URL it
prints.

If the avatar sits on **Ready** while you talk, the agent is not publishing —
check `run_agent.py` is actually running and that `logs/ui_state.json` is being
written. The face degrades to idle by design; it never blocks the demo.

---

## The three acts

Hold **SPACE** while you speak. Release to send. Everything below is said out
loud; the bracketed lines are what should happen.

### ACT 1 — it's alive (45s)

It opens on its own with **"Hey! I am Groot!"** and tells you to hold the
spacebar. Then:

> "What gestures can you do?"
> "Take a bow."           *(it should keep talking while the arm moves)*
> "Improvise a tiny robot dance."
> "What's on the table?"  *(the `look` tool — open-vocabulary; it names whatever
> is actually there, not just the five it can pick. Put something unexpected on
> the table for this one.)*

**Table:** the demo objects on their marked spots. **Point:** personality, that
speech and motion overlap, and that it can see the scene before you ask it to
touch anything.

### ACT 2 — a clean pick (45s)

**Table:** ONE red block, clearly on ONE marked spot. Clear the other spots.

> "Pick up the red block."

Expected: it states a confidence number, picks, and then **tells you whether it
actually got it**. If the grasp misses, it says so — that is a *win*, not a
failure. Point at the screen: G5 caught it.

**Screen:** confidence bar above the 60% line, all five gates green.

### ACT 3 — the gates (90s, this is the pitch)

**3a — ambiguity.** Two red blocks, on two different spots.

> "Pick up the red block."

Expected: **"Which one?"** — it names the two spots. Answer out loud
("the front-left one"). It resolves and picks.

> Say while it thinks: *"It won't guess. Guessing is how robots hurt people."*

**3b — low confidence + stand-down.** One object placed BETWEEN two spots, or a
partly-occluded one.

> "Pick up the charger."

Expected: it states a number below 60%, asks for approval — **and you say
nothing.** After ten seconds it stands down on its own. The arm never moves.

> Say: *"Nobody answered, so it did nothing. That timer is in Python, not in the
> prompt. You cannot talk it out of standing down."*

**Screen:** headline **STOOD DOWN — NO ANSWER**, G4 red, `moved: false`.

**3c — the honest no.** 

> "Pick up the unicorn."

Expected: "I can't see a unicorn." G1, no motion.

---

## Closing line

> "Every one of those decisions is in the log on the right. Which gate fired,
> what it was sure of, what the human said, and what it did. That is the audit
> trail — and none of it is in the prompt. The model can only speak the question
> and relay the answer; the gates are ordinary Python it cannot reach."

---

## When it goes wrong

| symptom | do this |
|---|---|
| **Gemini 429 mid-demo** | It says "my eyes are buffering". Switch to the replay dashboard and narrate: `python scripts/run_dashboard.py --replay` |
| **Venue wifi dies** | Phone hotspot. If that fails: replay dashboard + Act 1 (gestures need no network). |
| **Arm won't connect** | Power-cycle the arm, replug USB, `python tests/smoke_01_ports.py`. Worst case run `run_agent.py --no-motion` — personality and gates still demo. |
| **Arm behaves oddly / anything scary** | **Ctrl-C** → it freezes and holds → choose `[s]` to return to where the move started. Second Ctrl-C is a hard abort and leaves it exactly where it is. |
| **Grasp misses** | Say nothing and let it verify — it will announce the miss. That *is* the demo. |
| **It asks "which one?" too often** | Objects are landing between spots. Move them onto the marks. Persistent: lower `ARMANI_ASSIGNMENT_MARGIN_PX` to 40 in `.env` and restart. |
| **It says "not on a marked spot"** | Object is too far from every mark, or the camera moved. Re-run `python scripts/define_zones.py` (2 min). |
| **Datasets missing** | `python scripts/backup_datasets.py --list`, then copy a timestamped directory back, or point `ARMANI_GESTURE_ROOT` / `ARMANI_PICK_ROOT` at it. |

**The kill switch is the answer to every "is this safe?" question from the
audience.** Ctrl-C freezes and *asks you* what to do. It never auto-drives.

---

## Code freeze

Tagged `demo-freeze`. After that tag, the only acceptable changes are:

- a `.env` threshold (`ARMANI_ASSIGNMENT_MARGIN_PX`, `ARMANI_CONF_APPROVAL`)
- re-running `define_zones.py` if the camera moved
- re-recording a pick macro that stopped working

**Do not refactor, do not "quickly fix" a gate, do not upgrade a dependency.**
If something is broken enough to need code, it is broken enough to cut from the
demo. Rehearse the three acts until they are boring.

```bash
git status          # must be clean
git describe --tags # must say demo-freeze
```

## Backup video

Record one clean run of all three acts (phone on a tripod, screen + arm in
frame) and keep it on the presenting laptop. If the hardware dies entirely, the
video plus the replay dashboard still tells the whole story.
