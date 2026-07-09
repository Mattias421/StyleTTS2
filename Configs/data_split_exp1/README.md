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

## Results

All metrics below use the final configured checkpoint and 2,000 paired
validation utterances. Lower is better. Validation loss is the value logged at
the end of the final training epoch.

| Split | Base MCD | CDE MCD | Base log-F0 RMSE | CDE log-F0 RMSE | Base validation loss | CDE validation loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1m | 5.2666 +/- 0.9403 | 5.3798 +/- 0.9091 | 0.4148 +/- 0.1586 | 0.4003 +/- 0.1473 | 0.252 | 0.281 |
| 5m | 5.1632 +/- 0.9125 | 5.1710 +/- 0.9212 | 0.4059 +/- 0.1558 | 0.4055 +/- 0.1588 | 0.233 | 0.245 |
| 15m | 5.0831 +/- 0.9169 | 5.3715 +/- 0.9606 | 0.4139 +/- 0.1584 | 0.4036 +/- 0.1529 | 0.231 | 0.246 |
| 30m | 5.0968 +/- 0.9723 | 5.0272 +/- 0.9526 | 0.4027 +/- 0.1611 | 0.3844 +/- 0.1511 | 0.228 | 0.232 |
| 60m | 5.2316 +/- 0.9664 | 5.0469 +/- 0.9493 | 0.4245 +/- 0.1636 | 0.3934 +/- 0.1538 | 0.227 | 0.232 |

CDE improves log-F0 RMSE for four of five splits and is effectively tied at
5m. Its MCD is better at 30m and 60m, but worse at 1m, 5m, and especially 15m.
The final validation loss generally decreases as more data is added, while the
generation metrics are not monotonic across these short fixed-budget runs.
