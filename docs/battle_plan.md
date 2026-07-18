# SO-101 Voice Robot — 20-Hour Battle Plan
**Verdict: ACHIEVABLE. Zero training required for the core demo. Your Colab credits are optional insurance, not the plan.**

Date: 2026-07-18 · Hardware: SO-101 leader+follower (teleop working ✅) + Logitech C920 · Keys: OpenAI ✅ Anthropic ✅ Gemini ❌ (get free AI Studio key — 5 minutes, do it first)

---

## 1. Direct answer to your question

**"Is it possible that we don't train and it's working?" — YES, and it's the LOWER-risk path.**

Every capability in your vision maps to a no-training solution:

| Your vision | How (no training) |
|---|---|
| "Hi, hello, bow, move this side" + funny replies | Record gestures via teleop once, replay them (`lerobot-record` → `lerobot-replay`). Voice brain = OpenAI Realtime API with function calling. |
| "Pick that object" among 4–5 objects | Gemini points at the object in the C920 frame → homography converts pixel → table coordinates → IK top-down grasp. Google published the exact SO-101 sample repo. |
| "Which object do you mean?" | Detector finds 2 matches → agent asks. KnowNo-style multiple-choice logprob gate (CoRL 2023 Best Student Paper mechanism). |
| "30% confidence, need approval" | Composite of detector score + ambiguity margin + IK reachability. Below threshold → speak the % → wait for verbal "yes" → 10s timeout = stand down (fail-closed). |
| Funny in realtime | `gpt-realtime-2.1` (GA): ~500–800ms voice-to-voice, **async function calling = it banters WHILE the arm moves**. Personality is pure system prompt. |

Field evidence for why no-training wins: at the June 2025 LeRobot Worldwide Hackathon, SmolVLA fine-tune teams had Colab crash overnight and **never shipped**; winners were teams with one reliable skill + a thin voice/LLM wrapper. Also: ACT is NOT language-conditioned — "pick the banana vs the cup" would need one trained policy per object. Doesn't fit 20h. The VLM-grounding path handles named objects natively.

---

## 2. Architecture

```
 [Headset mic, push-to-talk spacebar]
        │ audio
        ▼
 OpenAI Realtime API (gpt-realtime-2.1)     ← personality lives in system prompt
   │ tool calls (async — keeps talking while arm moves)
   ├─ play_gesture(name)        → replay recorded episode (bow/wave/dance/nod/…)
   ├─ improvise_move(desc)      → Claude writes clamped joint keyframes (party trick)
   ├─ look_and_list()           → C920 frame → Gemini ER points + OWLv2 boxes/scores
   ├─ pick(object)              → TRUST GATES → homography → IK grasp → verify
   └─ stop() / home()           → always available, fail-closed
        │
 TRUST GATES (the startup):
   G1 seen?      OWLv2 top score < 0.35            → "I don't see a banana, chief."
   G2 ambiguous? 2+ candidates close / MCQA set>1   → "Two cups. Which one?"
   G3 reachable? IK solve + table polygon           → "That's outside my reach."
   G4 confident? composite < 0.6                    → "I'm 34% on this. Approve?"
                                                       10s silence → stand down
   G5 verified?  post-grasp frame → success check   → "Got it. Verified. You're welcome."
        │
 [Dashboard screen: camera + boxes + confidence bars + which gate fired + decision log]
        │
 LeRobot SO101Follower.send_action() @ your control loop  (or phosphobot REST fallback)
```

---

## 3. The stack (decided — stop deliberating, start building)

- **Voice/brain:** OpenAI Agents SDK realtime (`pip install openai-agents`), WebSocket, `gpt-realtime-2.1`. You have the key. Push-to-talk (`turn_detection: null` + spacebar), wired headset mic — **never the C920 mic, never open mic in the hall**. Cost: single-digit dollars for the whole day.
- **Grounding:** clone **google-gemini/robotics-pointing-sample** (Apache-2.0) — SO-101 + USB cam + Gemini pointing + ChArUco homography + IK. It POINTS/moves-to; you extend with descend→close-gripper→lift. Model: `gemini-robotics-er-1.6-preview` (free-tier AI Studio key works; fallback `gemini-3-flash` boxes or 1.5-preview).
- **Local scores (honest confidence):** OWLv2 (`google/owlv2-base-patch16-ensemble`, ~10 lines of transformers) gives real per-query probabilities + top-2 margin. Runs on laptop.
- **Motion:** LeRobot v0.6.0 (`pip install lerobot[feetech]`). Gestures = `lerobot-record`/`lerobot-replay` or a python wrapper replaying episode actions via `send_action`. IK = the sample repo's, or LeRobot `RobotKinematics` (Placo + `so101_new_calib.urdf`), or **phosphobot** REST `/move/absolute` as plan B.
- **Do NOT use GPT-4o/4.1 for object localization** — documented to fail at bounding boxes. OpenAI = dialogue/personality; Gemini/OWLv2 = pixels.
- **Cyberwave:** platform not needed (you're not their event). But their `nl_arm_controller` example is **MIT and portable** — steal `planner.py` (Claude→JSON joint moves, few-shot, defensive parser) for `improvise_move()`, and `motion.py`'s safety pattern (≤8 actions, ≤5s durations, per-joint clamps ±90°/±60°, exception→home, 20Hz interpolation). Their own demo prompts are "wave at the audience" and "do a small bow" — your exact feature, validated.

---

## 4. Grasp fallback ladder (the demo NEVER dies)

- **Plan A — true pixel-guided grasp:** Gemini point → homography → IK hover 10cm → descend → close → lift. Objects on a marked table region, top-down-graspable shapes (cube, banana, small cup, plush, marker).
- **Plan B — phosphobot REST IK:** if Placo/IK fights you >2 hours, `POST /move/absolute` with the same (x,y,z).
- **Plan C — taught poses (guaranteed):** 5 marked zones on the table, one recorded pick episode per zone. VLM still chooses WHICH zone (real perception!), replay does the grasp. Judges cannot tell the difference in wow. **Decide A→B→C by T+9. No sunk-cost heroics.**

---

## 5. Hour-by-hour (T = now; tracks run in parallel)

| Time | Track A — Motion | Track B — Voice brain | Track C — Perception/Trust |
|---|---|---|---|
| T+0–1 | **Lock the rig:** tape camera + table, mark 5 object zones + workspace polygon, verify calibration, `chmod 666 /dev/ttyACM*` | Get Gemini AI Studio key. `pip install openai-agents`. Print ChArUco board. | — |
| T+1–3 | Record 8 gestures (bow, wave, dance, nod-yes, shake-no, look-around, celebrate, sad-droop). Build `play_gesture()` replay wrapper. | Realtime skeleton: PTT, persona prompt, dummy tools, headset mic. | Clone pointing-sample; Gemini points at your 5 objects on C920 frames. |
| T+3–4 | **⭐ MILESTONE 1: talking, joking, bowing robot. Minimum demo SECURED by hour 4.** | ← integrate | Homography calibration (ChArUco, or jog gripper tip to 6 spots + `cv2.findHomography`). |
| T+5–9 | Extend point→grasp (hover/descend/close/lift). Gripper = just another joint in `send_action`. | `improvise_move()`: Claude→clamped keyframes (steal Cyberwave planner.py pattern). | OWLv2 on the 5 objects; scores + top-2 margin working. |
| T+9 | **DECISION GATE: Plan A working? Else switch B, else C. 30-min decision, no debate.** | | |
| T+9–12 | — | Wire trust gates into agent: clarify dialogue, approval flow, 10s fail-closed timeout, KnowNo MCQA logprob check (OpenAI API exposes logprobs). | IK reachability + polygon check; composite confidence: `0.5·det + 0.2·margin + 0.2·ik + 0.1·vlm_self_report` (tune on ~20 scripted trials). |
| T+12–14 | **Dashboard:** OpenCV/browser window — live camera, boxes/points, confidence bars, gate log. VCs watch this screen, not your code. | | Post-grasp verification: fresh frame → "is object in gripper?" → announce result. |
| T+14–16 | Full integration. Failure comedy lines. Kill switch visible. Test on phone hotspot. | | |
| T+16–18 | **Rehearse the 3-act script ×5. Record a clean backup video. FREEZE CODE.** | | |
| T+18–20 | Pitch deck + slack buffer (you will need it). | | |
| Overnight (optional) | ACT on Colab A100: 50 episodes, one task, ~1.5–2.5h, ~$3 → **roadmap slide only, never the live demo path.** Checkpoint to Hub (Colab dies overnight — it killed teams in June 2025). | | |

**Team split:** 3 people = one per track. 2 people = A+B converge at hour 4, second person owns C throughout.

---

## 6. Demo script (3 acts, ~4 minutes)

1. **Personality (60s):** banter, "take a bow," "do a happy dance you've never done before" → improvised move. Hook set.
2. **Competence (60s):** "pick up the banana" → dashboard shows detection, robot says "Banana. 92%. Easy money." → grasps → verifies → quips.
3. **Trust (90s) — the money act:**
   - "Hand me the cup" (two cups on table) → "I see two cups — the red one or the white one?" → "red" → executes.
   - Object placed at workspace edge → "I can reach it, but I'm at 34%. That's a coin flip with worse odds. Want me to try?" → "yes" → careful attempt. Then once WITHOUT approval → 10 seconds → "No approval, no action. I don't gamble with your table." → stands down.
4. **Close on the dashboard:** "Everyone builds robots that act. We built one that knows when *not* to."

Failure insurance: if a grasp fails live, the robot owns it — "That's on me. Recalibrating my ego." → retry once. A funny recovery beats a fake success.

---

## 7. Cut list (discipline = survival)

CUT: open mic (PTT only) · arbitrary objects (5 fixed, top-down graspable) · multi-step task chains · SmolVLA/GR00T/pi0 in the demo path (VRAM + crash risk; ACT needs one-policy-per-object for naming — dead end here) · full hand-eye calibration (plane homography subsumes it; a forum user lost days to `calibrateHandEye` being 0.5m off) · wake words · web/mobile app · simulation.

---

## 8. Risk register

| Risk | Counter |
|---|---|
| Noisy hall kills STT | Wired headset + PTT. `gpt-realtime-2.1` is noise-hardened, but PTT anyway. |
| Venue wifi dies | Phone hotspot, pre-tested. Fallback cascade: faster-whisper local + gpt-4o-mini + TTS (~1.5s, survives bad internet). |
| Camera bumped → homography off | Tape everything. Keep ChArUco board handy — recal is 5 min. Check alignment before demo. |
| Gemini robotics preview rate-limited | Fallback: standard Gemini flash boxes, or OWLv2-only (local, no API). |
| Servo flake / overheat | Don't hold torqued poses; power-cycle clears lockouts; recalibrate BOTH arms after any crash; keep `--robot.id` stable. |
| Demo-day catastrophe | Backup video recorded at T+16. Non-negotiable. |

---

## 9. VC pitch skeleton

- **Problem:** Robot arms hit $100–300 (SO-101, huge LeRobot community — thousands of builders). Everyone demos robots that act on command. Nobody solved *trust* — robots that guess wrong near humans are useless and unsafe.
- **Insight:** The unlock isn't a better policy. It's calibrated interaction: ground what you see → clarify ambiguity → quantify confidence → gate on approval → verify your own work. Fail-closed by default.
- **Product:** The trust/interaction layer for embodied AI — an SDK any robot embeds. Tonight's demo = reference implementation on $300 hardware.
- **Research lineage (credibility):** We productized the "Robots That Ask For Help" thread — KnowNo (CoRL 2023 **Best Student Paper**, conformal prediction), SayCan affordances, Inner Monologue verification — in 20 hours. Roadmap: conformal calibration, VLA token-uncertainty integration (Ask-Before-You-Act, RSS 2025), fleet audit logs.
- **Market validation:** Cyberwave raised **€7M (Oct 2025)** for adjacent robot-platform tooling; NVIDIA/HF/Google all shipping SO-101-compatible stacks. The interaction/trust wedge is open.
- **Close:** "Every car ships with seatbelts. Every robot will ship with a trust layer."

---

## 10. Command + link pack

```bash
# Setup
pip install "lerobot[feetech]" openai-agents transformers google-genai
sudo chmod 666 /dev/ttyACM*          # Linux serial perms

# Gestures (record once, replay forever)
lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=$HF_USER/gestures --dataset.num_episodes=8 --dataset.single_task="gesture"
lerobot-replay --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --dataset.repo_id=$HF_USER/gestures --dataset.episode=0

# Direct joint control (python)
# from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
# robot.send_action({"shoulder_pan.pos": ..., "gripper.pos": ...})  # normalized -100..100

# C920: run 640x480@30 — 1080p chokes bandwidth for no benefit
```

Key links: [robotics-pointing-sample](https://github.com/google-gemini/robotics-pointing-sample) · [Gemini robotics docs](https://ai.google.dev/gemini-api/docs/robotics-overview) · [OpenAI Agents SDK realtime](https://openai.github.io/openai-agents-python/realtime/quickstart/) · [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101) · [il_robots (record/replay)](https://huggingface.co/docs/lerobot/il_robots) · [phone_teleop (RobotKinematics/IK)](https://huggingface.co/docs/lerobot/phone_teleop) · [SO-101 URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf) · [OWLv2](https://huggingface.co/docs/transformers/main/en/model_doc/owlv2) · [phosphobot](https://github.com/phospho-app/phosphobot) · [lerobot-kinematics](https://github.com/box2ai-robotics/lerobot-kinematics) · [Cyberwave nl_arm_controller (MIT, steal planner/motion patterns)](https://github.com/cyberwave-os/cyberwave-python) · [KnowNo](https://robot-help.github.io/) · [SayCan](https://say-can.github.io/) · [Inner Monologue](https://innermonologue.github.io/) · [Reachy Mini conversation app (architecture to copy: tool dispatcher + motion thread + idle breathing)](https://github.com/pollen-robotics/reachy_mini_conversation_app) · [GLaDOS persona template](https://github.com/dnhkng/GLaDOS) · [LeCopain voice+ACT prizewinner blueprint](https://github.com/alexcbb/LeCopain) · [LeRobot hackathon field report](https://kamathrobotics.com/lerobot-worldwide-hackathon)
