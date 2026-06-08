# Data Split Experiment Matrix

This directory contains the split experiment configs for issue `CDE-71`.

- Splits: `1m`, `5m`, `15m`, `30m`, `60m`
- Base runs: 5 Mimas configs
- CDE runs: 5 Phoebe configs
- CDE settings: `hidden_channels: 256`, `num_layers: 6`, `dt: 0.5`
- Epoch schedules:
  - `1m`: `epochs=150`, `diff_epoch=30`, `joint_epoch=200`
  - `5m`: `epochs=30`, `diff_epoch=6`, `joint_epoch=200`
  - `15m`: `epochs=10`, `diff_epoch=4`, `joint_epoch=200`
  - `30m`: `epochs=5`, `diff_epoch=2`, `joint_epoch=200`
  - `60m`: `epochs=3`, `diff_epoch=2`, `joint_epoch=200`

Validation synthesis and MCD/log-F0 evaluation use:

```bash
# Run from the issue workspace on each named host.
scripts/run_data_split_exp1_eval.sh base 0  # mimas
scripts/run_data_split_exp1_eval.sh cde 0   # phoebe
```

The queue is resumable and waits for each training run's successful exit marker
and final checkpoint before generating the 2,000 paired validation samples.
