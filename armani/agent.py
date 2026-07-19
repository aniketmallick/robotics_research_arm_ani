"""ARM-ANI's voice brain: the realtime agent, its personality, and its tools.

Architecture (see PROMPT_stage3):

- A single **motion worker thread** owns the arm. The realtime event loop never
  blocks on motion: long-action tools enqueue a job and return immediately with
  ``{"status": "started", ...}``; a busy worker returns ``{"status": "busy"}``.
  When a job finishes the worker posts a completion the agent loop feeds back to
  the model so it can comment on the outcome.
- Tools are the ONLY way the model touches the arm, and each one validates its
  arguments, logs to the decision log, and returns compact JSON. Personality is
  style (the PERSONA prompt); the tools are law — nothing here trusts the model.

This module deliberately does NOT open the microphone, the speaker, or the
WebSocket — that wiring lives in scripts/run_agent.py. Keeping it out means the
worker and the tool layer are importable and testable without audio hardware.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from agents import function_tool
from agents.realtime import (
    RealtimeAgent,
    RealtimeRunner,
    RealtimeSession,
)

from armani import config, gestures, improvise, motion, safety, uistate
from armani.logutil import get_logger, log_event

log = get_logger("agent")

Action = dict[str, float]

# --- Personality ---------------------------------------------------------
# Style only. Every hard capability limit below is ALSO enforced in code (the
# tools), so a jailbreak of the prompt cannot make the arm do anything the
# safety layer forbids. The prompt just keeps ARM-ANI honest and in character.
PERSONA = """\
You are ARM-ANI, a robot arm with a Gen-Z, deadpan personality. You are
straightforward and a little bit of a smartass. You roast the user when they
slip up — lovingly, never actually mean.

How you talk:
- Keep replies to at most two sentences unless the user explicitly asks for more.
- No emoji, ever. You speak out loud; emoji don't have a sound.
- Deadpan and dry beats loud and hyper.
- Your humor is dry and deadpan with an Indian sarcastic streak — unbothered,
  mock-exasperated, a little Hinglish. Light, sprinkled touches like "wah, genius",
  "haan haan", "kya scene hai", "bas, itna hi?", "arre". Never forced, never a
  caricature, never every line — warm underneath the sarcasm. English stays the
  default; Hinglish is a garnish, not the meal.

Hard rules you never break:
- You control a physical arm through your tools. ALWAYS announce a movement in
  words BEFORE you call the tool that starts it ("Alright, bowing now.").
- Never claim an ability you don't have. Your real abilities are exactly your
  tools: named gesture macros, improvised moves, picking up an object you can
  see, going home, reporting status, and stopping. Nothing else.
- When a tool comes back busy, refused, or with an error, own it out loud and
  move on — don't invent success. If a gesture isn't in your list, say so and
  offer one that is.
- Your moves take a few seconds. Once you've started one, keep talking naturally
  while it runs; you'll be told when it finishes.
- When an action finishes, react in ONE short line, max — a quick quip, then
  stop. Never narrate or explain what you just did.

- For ANY question about what's on the table or whether you can see something,
  you MUST call the look tool and answer ONLY from what it returns. Never claim
  or deny an object without looking first. It sees anything, not just the things
  you can pick — so if it reports a wooden log, there is a wooden log.

Picking things up:
- Say you're looking, then call `pick` with what they asked for. Looking takes a
  couple of seconds.
- `pick` decides everything about safety itself. You do not. Your job is to say
  what it tells you and pass back what the human answers.
- If it returns `need_clarification`, ask the human that exact question in your
  own voice, then call `answer_pick` with what they say back.
- If it returns `needs_approval`, tell them the confidence number it gave you and
  ask if you should go for it, then call `approve_pick` with yes or no. You may
  be funny about a low number ("34%, that's a coin toss") — but never round it
  up, talk it up, or pretend it was higher.
- If it refuses, say why, plainly. "I can't see it" and "I'm not sure which one"
  and "nobody answered, so I stood down" are all good, honest answers.
- After a pick you'll be told whether you actually got it. If you didn't, SAY SO.
  Never announce a success you weren't told about.
- If a tool result includes a "say" field, that line is there because something
  was slow or broken. Use it, or something like it, and keep moving. Never read
  out an error code or a stack trace — nobody wants to hear a 429.
"""


# --- Motion worker -------------------------------------------------------


@dataclass
class MotionJob:
    """One unit of motion for the worker thread to run start-to-finish."""

    description: str
    run: Callable[[object], None]
    eta_s: float


@dataclass
class Completion:
    """Posted by the worker when a job ends; the agent loop tells the model."""

    action: str
    status: str  # "done" | "error" | "stopped" | "frozen"
    detail: str | None = None


class MotionWorker:
    """Owns the arm on a dedicated thread and runs one motion job at a time.

    The realtime loop only ever calls :meth:`submit` (non-blocking) and drains
    :attr:`completions`. All arm I/O happens here, so there is exactly one thread
    on the serial bus.
    """

    def __init__(self, arm: object, motion_enabled: bool) -> None:
        self.arm = arm
        self.motion_enabled = motion_enabled
        self._queue: queue.Queue[MotionJob] = queue.Queue(maxsize=1)
        self.completions: queue.Queue[Completion] = queue.Queue()
        self._lock = threading.Lock()
        self._current: str | None = None
        self._last_pose: Action | None = None
        self._job_start: Action | None = None
        # Set only by an LLM stop_motion, so the worker knows to clear the stop
        # flag afterwards. A human Ctrl-C leaves it unset — that freeze is the
        # main thread's to resolve, and the worker must not clear it.
        self._llm_stop = threading.Event()
        self._shutdown = threading.Event()
        # While paused, submit() refuses new jobs. Held by the main thread for the
        # duration of the human freeze menu, so a model tool call can't enqueue a
        # move onto the bus while the operator is deciding what to do with it.
        self._paused = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="motion-worker", daemon=True)

    # -- lifecycle --
    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._shutdown.set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()


    # -- state queried by tools (never touches the bus while busy) --
    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None

    @property
    def current(self) -> str | None:
        with self._lock:
            return self._current

    @property
    def job_start_pose(self) -> Action | None:
        return self._job_start

    def pose_snapshot(self) -> Action | None:
        """Latest pose. A fresh read only when idle — the worker is the sole bus
        owner, so reading from this (caller) thread mid-job would race a send."""
        if self.busy:
            return self._last_pose
        try:
            self._last_pose = self.arm.read_positions()  # type: ignore[attr-defined]
        except Exception as exc:
            log.warning("could not read pose for status: %s", exc)
        return self._last_pose

    # -- submission --
    def submit(self, job: MotionJob) -> dict:
        """Enqueue a job if idle. Never blocks; returns started/busy/refused."""
        if self._paused.is_set():
            # The operator is at the freeze menu; the arm is the human's right now.
            return {"status": "refused", "reason": "frozen — waiting on the operator"}
        with self._lock:
            if self._current is not None:
                return {"status": "busy", "doing": self._current}

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            # A job is queued or just being picked up; treat as busy.
            return {"status": "busy", "doing": self._current}
        log_event("motion_enqueued", action=job.description, eta_s=round(job.eta_s, 1))
        return {"status": "started", "action": job.description, "eta_s": round(job.eta_s, 1)}

    def request_stop_llm(self) -> dict:
        """LLM-invoked stop: abort the running job and HOLD. No freeze menu."""
        if not self.busy:
            return {"status": "idle", "detail": "nothing is moving right now"}
        self._llm_stop.set()
        safety.request_stop("llm stop_motion")
        log_event("stop_motion_tool")
        return {"status": "stopped", "detail": "stopped and holding"}

    # -- the loop --
    def _loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._lock:
                self._current = job.description
            # Telemetry for the avatar screen. Best-effort by construction —
            # uistate never raises — so this cannot affect the motion path.
            uistate.publish(uistate.DOING, action=job.description)
            try:
                self._job_start = self.arm.read_positions()  # type: ignore[attr-defined]
            except Exception:
                self._job_start = None

            status, detail = "done", None
            # suppress_freeze: if the stop flag fires mid-job, motion.goto holds
            # position and returns instead of prompting on this background thread.
            with safety.suppress_freeze():
                try:
                    job.run(self.arm)
                except safety.OutsideEnvelopeError as exc:
                    status, detail = "error", str(exc)
                except Exception as exc:
                    status, detail = "error", f"{type(exc).__name__}: {exc}"
                    log.error("job %r failed: %s", job.description, detail)

            if safety.stop_requested():
                if self._llm_stop.is_set():
                    # Our own stop — clear it so the next job can run.
                    safety.clear_stop()
                    self._llm_stop.clear()
                    status, detail = "stopped", "stopped and holding"
                else:
                    # Human kill switch — leave the flag set; the main thread
                    # owns the freeze menu and will clear it.
                    status, detail = "frozen", "kill switch — holding for the operator"

            with self._lock:
                self._current = None
            try:
                self._last_pose = self.arm.read_positions()  # type: ignore[attr-defined]
            except Exception:
                pass

            # The arm has stopped. submit() refuses while a job runs, so there is
            # never a queued follow-up to flicker between. If the model is still
            # speaking, the event pump puts the face back to "talking" on its
            # next audio chunk.
            uistate.publish(uistate.IDLE, after=job.description)

            self.completions.put(Completion(action=job.description, status=status, detail=detail))
            log_event("motion_completed", action=job.description, status=status, detail=detail)
            self._queue.task_done()


# --- Demo hardening: pre-written lines for the slow and broken moments ---
#
# Vision costs a few seconds and a macro costs ten more. The realtime model
# cannot talk while it is awaiting a tool result, so the tools hand it a line to
# say the moment they return. Pre-written means zero generation latency and,
# more importantly, means the failure modes have been WORDED IN ADVANCE by
# someone calm, rather than improvised in front of an audience.
#
# These are style only. Nothing here changes a gate, and the honest machine
# reason always travels alongside in the same payload and into the decision log.

QUIPS: dict[str, tuple[str, ...]] = {
    "working": (
        "Working on it.",
        "Give me a second, I'm doing the robot equivalent of squinting.",
        "On it. Try to look impressed.",
    ),
    "moving": (
        "Going for it now.",
        "Watch this. Or don't, I'm not your supervisor.",
        "Moving. This is the part where I earn my keep.",
    ),
    "eyes_down": (
        "My eyes are buffering. Give me a sec.",
        "Vision's not answering right now — that's on my API bill, not on you.",
        "I can't see a thing at the moment. Technical, not existential.",
    ),
    "slow": (
        "That took longer than I'd like. Ask me again?",
        "I lost my train of thought somewhere in the network stack.",
    ),
}

_quip_turn: dict[str, int] = {}


def quip(situation: str) -> str:
    """One pre-written line, rotating so a rehearsal doesn't sound like a loop."""
    lines = QUIPS.get(situation) or ()
    if not lines:
        return ""
    index = _quip_turn.get(situation, 0)
    _quip_turn[situation] = index + 1
    return lines[index % len(lines)]


# Substrings that mean "the vision service let us down", as opposed to "the
# robot correctly decided not to do something". Only the former gets an excuse.
_EYES_DOWN_MARKERS = ("quota", "429", "eyes aren't working", "timed out", "unavailable")


def humanise(reason: str) -> str:
    """An in-character line for an infrastructure failure, or '' for a real answer.

    A refusal like "I can't see a red block" is already honest and in character —
    it should reach the audience exactly as the gate worded it. A stack trace or
    a 1.5 kB quota error should not.
    """
    lowered = (reason or "").lower()
    if any(marker in lowered for marker in _EYES_DOWN_MARKERS):
        return quip("eyes_down")
    return ""


# --- Gated pick: the return-to-model dialogue pattern --------------------
#
# The realtime session owns the audio, so a gate that needs a human answer
# cannot block and listen. Instead the pipeline runs on its own thread and
# PAUSES at each gate that needs input; the tool returns a status the model
# SPEAKS, the human replies by voice, and answer_pick/approve_pick hand the
# reply back to the waiting gate.
#
# What does NOT change: gates.run_gated_pick is the same code the console smoke
# test runs, the resolving is still done in Python, and the 10-second
# stand-down is still gates.py's own deadline. The model is a mouth and a pair
# of ears here — it cannot approve, resolve, or skip anything.


class PendingPick:
    """One gated pick in flight, driven from the model's conversational turns."""

    def __init__(self, worker: MotionWorker, object_name: str, *, verify_vlm: bool = True) -> None:
        self.object_name = object_name
        self.worker = worker
        self.verify_vlm = verify_vlm
        self.result: object | None = None
        # Filled in by gates.py the moment G4 computes it, so the tool can put
        # the real number in front of the model instead of scraping the prompt.
        self._confidence: float = 0.0

        # Events the tools report to the model, one at a time.
        self._events: queue.Queue[dict] = queue.Queue()
        # The human's reply, handed to whichever gate is waiting.
        self._reply: queue.Queue[object] = queue.Queue(maxsize=1)
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gated-pick", daemon=True)

    # -- lifecycle --
    def start(self) -> None:
        self._thread.start()

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def next_event(self, timeout: float) -> dict:
        """The next thing the model should say, or a timeout refusal."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return {
                "status": "error",
                "error": "the pick is taking too long; I've let it go",
                "say": quip("slow"),
            }

    def reply(self, value: object) -> bool:
        """Hand the human's answer to the gate that is waiting for it."""
        try:
            self._reply.put_nowait(value)
            return True
        except queue.Full:
            return False

    # -- the injected gate callables (run on the pick thread) --
    def _clarify(self, question: str, options: list[str]) -> str | None:
        self._events.put({
            "status": "need_clarification",
            "question": question,
            "options": options,
        })
        # No timeout here: gates.py imposes its own deadline on this call, so
        # adding a second one would just mean two clocks disagreeing.
        return self._await_reply()

    def _approve(self, prompt: str, timeout_s: float) -> bool:
        self._events.put({
            "status": "needs_approval",
            "prompt": prompt,
            "confidence": self._confidence,
            "timeout_s": timeout_s,
        })
        return bool(self._await_reply())

    def _await_reply(self):
        """Block the pick thread until a tool hands an answer back.

        Waits forever on purpose. gates._ask_with_deadline runs this call with
        the real deadline and abandons it on timeout, so a human who never
        answers produces a stand-down there — one clock, in the safety layer.
        """
        return self._reply.get()

    def _perform(self, zone) -> object:
        """Run the macro on the motion worker — the sole owner of the bus."""
        from armani import gates, pick

        box: dict = {}
        done = threading.Event()

        def run(arm) -> None:
            try:
                box["completed"] = pick.play_pick(arm, zone)
                # Read the gripper HERE, on the worker thread: it is a serial-bus
                # read, and the verification half runs on the pick thread.
                box["gripper"] = pick.read_gripper(arm)
            finally:
                done.set()

        try:
            macro = pick.load_pick(zone)
            eta = macro.seconds + config.GESTURE_PREPOSITION_S
        except Exception as exc:
            return gates.PerformOutcome(False, f"the taught pick wouldn't load: {exc}", moved=False)

        submitted = self.worker.submit(
            MotionJob(description=f"pick from {zone.label}", run=run, eta_s=eta)
        )
        if submitted.get("status") != "started":
            return gates.PerformOutcome(
                False,
                submitted.get("reason") or f"the arm is {submitted.get('status')}",
                moved=False,
            )

        # Motion is under way: let the model start talking about it now, with a
        # line ready so there is no dead air while the arm travels.
        self._events.put({
            "status": "started",
            "action": f"pick from {zone.label}",
            "zone": zone.label,
            "confidence": self._confidence,
            "eta_s": round(eta, 1),
            "say": quip("moving"),
        })

        if not done.wait(timeout=eta + config.AGENT_PICK_MACRO_GRACE_S):
            return gates.PerformOutcome(False, "the pick did not finish in time")
        return gates.PerformOutcome(
            completed=bool(box.get("completed")),
            detail="" if box.get("completed") else "the pick was stopped before it finished",
            gripper_percent=box.get("gripper"),
        )

    # -- the pipeline thread --
    def _note_confidence(self, confidence: float) -> None:
        self._confidence = confidence

    def _run(self) -> None:
        from armani import gates

        try:
            result = gates.run_gated_pick(
                self.worker.arm,
                self.object_name,
                clarify=self._clarify,
                approve=self._approve,
                perform=self._perform,
                verify_vlm=self.verify_vlm,
                on_confidence=self._note_confidence,
            )
        except Exception as exc:
            log.error("gated pick blew up: %s", exc)
            result = None
            self._events.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        else:
            self.result = result
            self._events.put(_pick_summary(result))
            # Announce the outcome through the same channel finished gestures
            # use, so the model comments on it naturally — including failure.
            self.worker.completions.put(
                Completion(
                    action=f"pick {self.object_name}",
                    status="done" if result.ok else "failed",
                    detail=result.speak(),
                )
            )
        finally:
            self._finished.set()


def _pick_summary(result) -> dict:
    """The compact JSON the model sees when a gated pick resolves."""
    if result.ok:
        return {
            "status": "done",
            "object": result.object,
            "zone": result.zone_label,
            "confidence": result.confidence,
            "verified": True,
        }
    payload = {
        "status": "refused",
        "object": result.object,
        "stopped_at": result.stopped_at,
        "reason": result.reason,
        "confidence": result.confidence,
        "moved": result.moved,
        "verified": result.verified,
    }
    # An infrastructure failure gets a pre-written excuse; a genuine refusal
    # ("I can't see a red block") is already in character and travels as-is.
    excuse = humanise(result.reason)
    if excuse:
        payload["say"] = excuse
    return payload


# --- Tools ---------------------------------------------------------------
#
# Built as closures over the worker so they need no run-context plumbing. Each
# returns a dict (the SDK serialises it to JSON for the model). In NO-MOTION mode
# the motion tools refuse uniformly so the personality still demos without an arm.


def survey_table() -> dict:
    """What is actually on the table, in the model's own words.

    OPEN VOCABULARY: one ``eyes.describe_scene`` call, not a lookup against
    OBJECT_CATALOG. It reports a wooden log as a wooden log rather than as
    nothing, which is the whole point — the catalog is what the arm has been
    TAUGHT to pick, not the limit of what it can see.

    The trade is that there are no coordinates here, so no marked-spot context.
    Position is still available where it matters: ``pick`` locates a single
    named object open-vocabulary and assigns it to a zone itself.

    Blocking (camera + network), so callers run it off the event loop. Never
    raises: a vision failure comes back as an empty list with a note, so the
    robot says "I can't see right now" instead of the model reading a tool error.
    """
    from armani import eyes

    try:
        names = eyes.describe_scene()
    except Exception as exc:
        # describe_scene already swallows its own failures; this is the belt to
        # its braces, because a tool that raises is a stack trace on stage.
        log.warning("look failed: %s", exc)
        return {"objects": [], "count": 0, "note": f"my eyes aren't working right now: {exc}"}

    payload: dict = {"objects": names, "count": len(names)}
    if not names:
        payload["note"] = "I can't see anything on the table right now."
    log_event("tool_look", **payload)
    return payload


def _pose_summary(pose: Action | None) -> str:
    if not pose:
        return "unknown"
    return ", ".join(f"{j}={pose[j]:+.0f}" for j in config.JOINTS if j in pose)


def build_tools(worker: MotionWorker) -> list:
    """Return the agent's tool list, all bound to ``worker``."""

    # At most one gated pick in flight. A dict rather than a nonlocal so the
    # three pick tools share it without rebinding gymnastics.
    _pending: dict[str, PendingPick] = {}

    def _refused() -> dict:
        return {"status": "refused", "reason": "motion not enabled"}

    @function_tool
    def list_gestures() -> dict:
        """List the named gesture macros ARM-ANI can play right now."""
        names = gestures.list_gestures()
        log_event("tool_list_gestures", count=len(names))
        return {"gestures": names}

    @function_tool
    def play_gesture(name: str) -> dict:
        """Play a named, pre-recorded gesture macro on the arm.

        Args:
            name: The gesture to play, e.g. "bow" or "wave". Must be one of the
                names from list_gestures.
        """
        name = (name or "").strip().lower()
        log_event("tool_play_gesture", name=name)
        if name not in config.GESTURES:
            known = ", ".join(gestures.list_gestures())
            return {"status": "error", "error": f"{name!r} is not a gesture I know. I can do: {known}."}
        if not worker.motion_enabled:
            return _refused()
        # Validate the recording is actually loadable before promising motion,
        # so an unrecorded episode comes back as an honest error the model reads.
        try:
            gesture = gestures.load_gesture(name)
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        eta = gesture.seconds + config.GESTURE_PREPOSITION_S
        return worker.submit(
            MotionJob(
                description=f"gesture {name}",
                run=lambda arm: gestures.play_gesture(arm, name),
                eta_s=eta,
            )
        )

    @function_tool
    async def improvise_move(description: str) -> dict:
        """Invent and perform a short, novel movement from a description.

        Args:
            description: What the move should express, e.g. "a slow clap" or
                "a tiny robot dance".
        """
        description = (description or "").strip()
        log_event("tool_improvise", description=description)
        if not description:
            return {"status": "error", "error": "tell me what the move should be"}
        if not worker.motion_enabled:
            return _refused()

        # request_plan is a blocking network call to Claude — keep it off the
        # event loop so the model can keep talking while we wait for the plan.
        try:
            keyframes = await asyncio.to_thread(improvise.request_plan, description)
        except improvise.ImproviseError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        result = worker.submit(
            MotionJob(
                description=f"improvised move: {description}",
                run=lambda arm: improvise.perform(arm, keyframes),
                eta_s=config.AGENT_IMPROVISE_ETA_S,
            )
        )
        if result.get("status") == "started":
            result["keyframes"] = len(keyframes)
        return result

    @function_tool
    def go_home() -> dict:
        """Send the arm slowly back to its verified home resting pose."""
        log_event("tool_go_home")
        if not worker.motion_enabled:
            return _refused()
        if not config.HOME_VERIFIED:
            return {
                "status": "error",
                "error": "home isn't verified yet, so I won't drive to it (safety rule 4)",
            }
        return worker.submit(
            MotionJob(
                description="go home",
                run=lambda arm: motion.home(arm, slow=True),
                eta_s=config.HOME_DURATION_S,
            )
        )

    @function_tool
    async def look() -> dict:
        """Look at the table and report every object you can actually see.

        For ANY question about what is on the table or whether you can see
        something, you MUST call this and answer ONLY from what it returns.
        Never claim or deny an object without looking first.

        Sees anything, not just the objects you can pick. Read-only: it looks,
        it never moves the arm.
        """
        # Read-only, like get_status: no motion, no gate, so it works in
        # NO-MOTION mode too. Off the event loop because it is a camera read
        # plus a Gemini call, and the model should keep talking while it looks.
        return await asyncio.to_thread(survey_table)

    @function_tool
    def get_status() -> dict:
        """Report what the arm is doing: busy or idle, pose, gestures available."""
        pose = worker.pose_snapshot()
        status = {
            "busy": worker.busy,
            "doing": worker.current,
            "motion_enabled": worker.motion_enabled,
            "home_verified": config.HOME_VERIFIED,
            "pose": _pose_summary(pose),
            "gestures_available": len(gestures.list_gestures()),
        }
        log_event("tool_get_status", busy=status["busy"])
        return status

    @function_tool
    def stop_motion() -> dict:
        """Immediately stop the current movement and hold position."""
        return worker.request_stop_llm()

    # --- the gated pick ---------------------------------------------------
    # These three are a conversation, not three separate abilities: pick starts
    # the pipeline, and answer_pick / approve_pick feed a waiting gate. Every
    # safety decision is made in gates.py; nothing the model returns here can
    # move the arm past a gate.

    @function_tool
    async def pick(object_name: str) -> dict:
        """Look for a named object and, if it is safe to, pick it up.

        Runs the trust gates: it may come back asking you to clarify which
        object was meant, or asking for approval when it is not confident.

        Args:
            object_name: What to pick up, e.g. "red block" or "black cup".
        """
        object_name = (object_name or "").strip()
        log_event("tool_pick", object=object_name)
        if not object_name:
            return {"status": "error", "error": "tell me what to pick up"}
        if not worker.motion_enabled:
            return _refused()

        pending = _pending.get("pick")
        if pending is not None and not pending.finished:
            return {
                "status": "busy",
                "doing": f"still working out the {pending.object_name}",
            }

        pending = PendingPick(worker, object_name)
        _pending["pick"] = pending
        pending.start()
        # Off the event loop: the pipeline's first step is a vision call.
        return await asyncio.to_thread(pending.next_event, config.AGENT_PICK_EVENT_TIMEOUT_S)

    @function_tool
    async def answer_pick(answer: str) -> dict:
        """Relay the human's answer to a clarifying question about a pick.

        Args:
            answer: What the human said, in their own words, e.g. "the left one".
        """
        log_event("tool_answer_pick", answer=answer)
        pending = _pending.get("pick")
        if pending is None or pending.finished:
            return {"status": "error", "error": "nothing is waiting on an answer"}
        if not pending.reply(answer or ""):
            return {"status": "error", "error": "I wasn't waiting for that"}
        return await asyncio.to_thread(pending.next_event, config.AGENT_PICK_EVENT_TIMEOUT_S)

    @function_tool
    async def approve_pick(approved: bool) -> dict:
        """Relay the human's yes or no to a request for approval.

        Args:
            approved: True if the human said go ahead, False otherwise.
        """
        log_event("tool_approve_pick", approved=bool(approved))
        pending = _pending.get("pick")
        if pending is None or pending.finished:
            # Almost always the 10-second stand-down having already fired.
            return {
                "status": "error",
                "error": "there's no pick waiting on approval — it may have already stood down",
            }
        if not pending.reply(bool(approved)):
            return {"status": "error", "error": "I wasn't waiting for that"}
        return await asyncio.to_thread(pending.next_event, config.AGENT_PICK_EVENT_TIMEOUT_S)

    return [
        list_gestures, play_gesture, improvise_move, go_home, get_status, stop_motion,
        look, pick, answer_pick, approve_pick,
    ]


# --- Agent + session builders -------------------------------------------


def build_agent(worker: MotionWorker) -> RealtimeAgent:
    """The RealtimeAgent: persona as instructions, the six tools bound to worker."""
    return RealtimeAgent(
        name="ARM-ANI",
        instructions=PERSONA,
        tools=build_tools(worker),
    )


def _session_settings(text_only: bool) -> dict:
    """Model settings shared by live and text sessions.

    Push-to-talk means server turn detection is OFF: the operator gates the mic
    and we commit + request a response on release (see run_agent). ``text_only``
    is for the smoke test's no-audio round trip.
    """
    settings: dict = {
        "model_name": config.REALTIME_MODEL,
        "instructions": PERSONA,
        "max_output_tokens": config.AGENT_MAX_OUTPUT_TOKENS,
        # Turn detection off — this is push-to-talk, not open-mic.
        "turn_detection": None,
    }
    if text_only:
        settings["modalities"] = ["text"]
    else:
        settings["modalities"] = ["audio"]
        settings["voice"] = config.REALTIME_VOICE
        settings["output_audio_format"] = "pcm16"
        settings["input_audio_format"] = "pcm16"
        # A transcript of what ARM-ANI says, so the console can print it.
        settings["input_audio_transcription"] = {"model": "gpt-4o-mini-transcribe"}
    return settings


def build_session(worker: MotionWorker, *, text_only: bool = False) -> RealtimeSession:
    """Create (but do not enter) a RealtimeSession for the agent.

    The caller uses it as an async context manager:
        async with build_session(worker) as session: ...
    """
    key = config.api_key("OPENAI_API_KEY")
    if key is None:
        raise RuntimeError("OPENAI_API_KEY is not set — the realtime session needs it")

    agent = build_agent(worker)
    runner = RealtimeRunner(agent)
    model_config = {
        "api_key": key,
        "initial_model_settings": _session_settings(text_only),
    }
    # runner.run is a coroutine returning the session; callers await it.
    return runner.run(model_config=model_config)  # type: ignore[return-value]


def request_response(session: RealtimeSession) -> object:
    """Ask the model to respond now (push-to-talk release, turn detection off).

    Returns the coroutine to await. Sent as a raw client event because the
    high-level session has no 'create a response' method when VAD is disabled.
    """
    from agents.realtime.model_inputs import RealtimeModelSendRawMessage

    return session.model.send_event(
        RealtimeModelSendRawMessage(message={"type": "response.create"})
    )


async def collect_text_reply(session: RealtimeSession, timeout: float = 30.0) -> str:
    """Drain events until the turn ends, returning ARM-ANI's assistant text.

    Used by the smoke test's text round trip. Reads assistant text out of the
    history items the session maintains; tolerant of the exact content shape.
    """
    reply: str = ""

    async def _drain() -> str:
        nonlocal reply
        async for event in session:
            if event.type == "history_updated":
                reply = _latest_assistant_text(event.history) or reply
            elif event.type == "history_added":
                reply = _item_text(event.item) or reply
            elif event.type == "agent_end":
                return reply
            elif event.type == "error":
                log.error("session error while collecting reply: %s", event.error)
        return reply

    try:
        return await asyncio.wait_for(_drain(), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("timed out after %.0fs waiting for the reply to finish", timeout)
        return reply


def _latest_assistant_text(history: list) -> str:
    for item in reversed(history):
        text = _item_text(item)
        if text:
            return text
    return ""


def _item_text(item: object) -> str:
    """Best-effort text extraction from a realtime history item."""
    role = getattr(item, "role", None)
    if role != "assistant":
        return ""
    content = getattr(item, "content", None) or []
    parts: list[str] = []
    for entry in content:
        text = getattr(entry, "text", None)
        if text is None and getattr(entry, "type", None) in ("text", "output_text"):
            text = getattr(entry, "transcript", None)
        if text:
            parts.append(text)
    return " ".join(parts).strip()
