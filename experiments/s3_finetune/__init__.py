"""Spike S3 — fine-tune SmolVLA on ONE in-domain task ("Pick up the red block").

Build + document the record -> fine-tune -> eval pipeline. NOT the demo path.
Data is collected in the demo `lerobot` env (calibrated teleop); fine-tuning runs
on a Colab GPU (or Mac-MPS, slowly); eval reuses the S2 runner in `lerobot-vla`,
with `armani.safety.clamp_action` still in the send path. No `armani/` demo logic
is touched.
"""
