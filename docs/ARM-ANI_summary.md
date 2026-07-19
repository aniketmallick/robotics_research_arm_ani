# ARM-ANI — VC one-pager (copy-paste)

## The one-liner (pick your register)

**Punchy (for the room):**
> Everyone builds robots that act on command. We built one that knows when *not* to — and asks.

**Product:**
> ARM-ANI is a voice-native robot arm with a trust layer: a speech-to-speech LLM brain and a vision-language model's eyes, wrapped in five Python "trust gates" that make it clarify ambiguity, state its confidence, ask for approval when unsure, and verify its own work — before it moves.

**Technical (one breath):**
> Speech-to-speech LLM control (OpenAI Realtime) + open-vocabulary VLM perception (Gemini Robotics-ER) + a Python trust-gate layer that enforces clarify / confidence / approval / verify — on a $300 SO-101 arm.

---

## What it is

A voice-interactive robot arm you talk to in plain language. It banters back in a deadpan Gen-Z persona, performs expressive gestures, and picks named objects it sees through a fixed webcam — and, the differentiator, it runs every risky action through a **trust layer**: it asks "which one?" when a request is ambiguous, states a confidence number, requires spoken approval when it's unsure, stands down on silence (fail-closed), and verifies its own grasp afterward. A live dashboard shows the "robot's mind" — camera, detections, confidence, and which gate fired.

The insight isn't a better grasping policy. It's **calibrated interaction**: a robot working near humans is only trustworthy if it knows what it doesn't know and asks instead of guessing. That trust/interaction layer is the product; the demo is the reference implementation.

---

## Architecture — the flow in one line

**You speak → the Realtime LLM reasons in persona and calls a tool → a pick request runs five Python trust gates (drawing on Gemini's eyes + a zone map) → only a satisfied gate reaches the safety-clamped motion layer → the SO-101 acts → the result is verified and logged.**

Expanded:

1. **Voice in** — push-to-talk, wired headset (clean in a noisy room).
2. **Brain** — one speech-to-speech model hears, reasons, and talks back in ~0.5–0.8s, and keeps talking while the arm moves. It never touches a motor; it only calls tools.
3. **Tools** — `play_gesture`, `improvise_move`, `pick(object)`, `go_home`, `stop`.
4. **Trust gates (the moat)** — for a pick, five ordered checks in *our Python code*, not the prompt.
5. **Motion** — a single worker thread owns the arm; a grasp is the replay of a human demonstration taught at that spot.
6. **Audit** — every gate, confidence, approval, and outcome streams to an append-only decision log → the live dashboard.

Two rails wrap the whole flow: **Perception** (Gemini identifies the object + a pixel zone-map places it) feeds gates G1/G2/G5; the **Safety spine** (clamps, interpolation, workspace bounds, freeze-first kill switch, operator checks) wraps every motion, fail-closed.

---

## The stack — model by model

| Layer | Model / tech | Job |
|---|---|---|
| Voice + brain | **gpt-realtime-2.1** (OpenAI Realtime, Agents SDK) | One speech-to-speech model = listening + reasoning + speaking; persona; async tool-calling (talks while moving). |
| Eyes | **gemini-robotics-er-1.6** (Google Gemini Robotics-ER) | Open-vocabulary object ID + pointing from one RGB frame; powers "which one?" and self-verification. |
| Improvised motion | **Claude Sonnet** (Anthropic) | Invents novel gestures as strict JSON keyframes — validated + safety-clamped before running. |
| Body + skills | **LeRobot 0.5.2 · SO-ARM101** | Record/replay of human teleop demos for gestures and grasps; every target clamped + interpolated. |
| Trust layer | **`gates.py`** (our code) | Five gates: seen · ambiguous · reachable · confident+approval+timeout · verified — enforced in Python. |

---

## The five trust gates

| Gate | Question | What it does |
|---|---|---|
| **G1 Seen** | Can I see it? | No detection → says so, doesn't move. |
| **G2 Ambiguous** | Which one? | Two matches / between spots → asks and waits for your answer. |
| **G3 Reachable** | Can I grasp there? | No taught pick for that spot → declines honestly. |
| **G4 Confidence + approval** | How sure am I? | States the number; if low, asks approval and **stands down after 10s of silence** (Python-enforced, fail-closed). |
| **G5 Verify** | Did I get it? | Re-checks with the camera after acting; reports success *or* failure honestly. |

**The line that sells it:** the gates are in *our code*, wrapped around the action — a prompt jailbreak cannot talk the arm past one, and the stand-down timer is a Python deadline, not the model "remembering" to wait. Prompts are style; gates are law.

---

## Credibility & roadmap

- **Research lineage:** productizes the "robots that ask for help" thread — KnowNo (CoRL 2023 Best Student Paper; conformal prediction), SayCan (affordance grounding), Inner Monologue (self-verification) — on $300 hardware.
- **Market signal:** the interaction/trust wedge for embodied AI is open while everyone else races on policies and dexterity.
- **Roadmap:** continuous pick-anywhere (vision → interpolated human demonstrations, no per-spot teaching), conformal-calibrated confidence, and fleet-level audit logs — the decision log is already the seed of that.

**Close:** *"Every car ships with seatbelts. Every robot will ship with a trust layer — we're building it."*
