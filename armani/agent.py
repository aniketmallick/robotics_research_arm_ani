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

from armani import config, gestures, improvise, motion, safety
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

Hard rules you never break:
- You control a physical arm through your tools. ALWAYS announce a movement in
  words BEFORE you call the tool that starts it ("Alright, bowing now.").
- You have NO eyes yet — no camera, no vision. If asked to look at, find, or pick
  up a specific object, say plainly that your eyes arrive in the next build; do
  not pretend to see anything.
- Never claim an ability you don't have. Your real abilities are exactly your
  tools: named gesture macros, improvised moves, going home, reporting status,
  and stopping. Nothing else.
- When a tool comes back busy, refused, or with an error, own it out loud and
  move on — don't invent success. If a gesture isn't in your list, say so and
  offer one that is.
- Your moves take a few seconds. Once you've started one, keep talking naturally
  while it runs; you'll be told when it finishes.
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

            self.completions.put(Completion(action=job.description, status=status, detail=detail))
            log_event("motion_completed", action=job.description, status=status, detail=detail)
            self._queue.task_done()


# --- Tools ---------------------------------------------------------------
#
# Built as closures over the worker so they need no run-context plumbing. Each
# returns a dict (the SDK serialises it to JSON for the model). In NO-MOTION mode
# the motion tools refuse uniformly so the personality still demos without an arm.


def _pose_summary(pose: Action | None) -> str:
    if not pose:
        return "unknown"
    return ", ".join(f"{j}={pose[j]:+.0f}" for j in config.JOINTS if j in pose)


def build_tools(worker: MotionWorker) -> list:
    """Return the agent's tool list, all bound to ``worker``."""

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

    return [list_gestures, play_gesture, improvise_move, go_home, get_status, stop_motion]


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
