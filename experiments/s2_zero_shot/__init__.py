"""Spike S2 — zero-shot SmolVLA baseline on the real SO-101 table.

Measures what an untuned generalist VLA (lerobot/smolvla_base) does on this
arm/camera over scored trials. NOT a demo path. Lives in its own parallel conda
env (`lerobot-vla`); the demo env and every file under `armani/` stay untouched.
Every predicted action passes through the policy-profile clamp before any send.
"""
