# ACT co-training experiment (cotrain_v1): training instructions

Paper: sim-and-real co-training of ACT on the Franka four-cube task (alpha sweep, sim-quantity sweep, real-only and
sim-only curves). Every model in this experiment must be trained the same way, so that the only difference between
two models is the variable in their name.

> ## ⚠️ HARDWARE AND BATCH SIZE ARE LOCKED
> **Train every model in this experiment on an NVIDIA RTX 6000 Ada Generation (48 GB) at batch size 40.**
> CIR2 and CIR1 both have this GPU. Owner's decision (17 Nov 2026): **CIR1 trains the 13 remaining cells**;
> CIR2 (done: S350_a090, S350_a100) stays on the door project.
> - No other GPU, machine or cloud. The paper states one hardware setup.
> - Do **not** change the batch size to fit memory or to speed up. A different batch changes samples per step,
>   so its 100k-step model is not comparable.
> - If the GPU is busy, **wait or share it**. Do not change the settings to make it fit.
> - One run at a time per machine: two at once ran out of RAM (62 GB) on both machines.
> - Exception: model 4 (real-only, 50 demos, the older c100 run) is batch 32. It is footnoted, not repeated.
> - **The protocol is set by this folder** (`make_cells.py`, `cells.txt`, `excl_*.txt`, `cells.json`). A run with any other
>   subset selection (for example CIR1's seeded-permutation S050_a090, 17 Nov) is **off-protocol**: it may be
>   reported as a side note, never as a sweep point.

## 1. Locked settings (the same for every model)

| Setting | Value |
|---|---|
| **GPU** | **NVIDIA RTX 6000 Ada Generation, 48 GB (CIR2)** |
| **Batch size** | **40** |
| Steps | 100,000; the **100k checkpoint** is the evaluated model, never "the best" |
| Dataset | `~/bc_data/lerobot/franka/cubes_20261014_30hz_real50_mimic376` (50 real + 376 Mimic = 426 episodes, 30 Hz) |
| Subsets | `--exclude-episodes` lists `excl_<cell>.txt` in this folder; sim subsets = first N Mimic demos by checklist order (nested); R25 = every 2nd real episode |
| Normalisation | stats of the full merged dataset (cameras: ImageNet stats), the same for every model |
| Policy | LeRobot 0.4.1 ACT, chunk 100, `n_action_steps` 100; inputs: 8-D joint state, 7-D EE pose, wrist + base cameras 640x480 |
| Optimiser | AdamW, lr 3e-5, backbone lr 1e-5, weight decay 1e-4, grad clip 10, no LR schedule |
| Seed | 1000 (training and sampler) |
| Augmentation | off |
| alpha | per-sample probability of drawing a sim frame (`train_weighted.py --alpha-sim`), set per cell in `cells.txt` only |

All of these come from `scripts/train_weighted.sh` (preset `real50_mimic376`) and `scripts/queue_cells.sh`.
Do not edit them for this experiment.

## 2. Safe to change (no effect on the trained model)

| Setting | Default | Note |
|---|---|---|
| `NUM_WORKERS` | 8 | data loading only. 16 was OOM-killed (RAM) on 9 Nov 2026; do not raise. |
| `SAVE_FREQ` | 20000 | resume points; deleted after the run, only 100k kept (disk) |
| `MAX_RETRY` | 3 | automatic resumes after a crash |
| `WANDB` | true | logging only. **`WANDB=false` on CIR1** (no wandb login: lerobot 0.4.1 forces online mode and `WANDB_MODE=offline` does not override it). Loss stays in the run `.log`. |
| Order in `cells.txt` | | reorder or `#` lines that have not started; re-read before each run |

## 3. Start or resume the queue

```bash
nvidia-smi        # must show the RTX 6000 Ada; the door project has GPU priority unless the owner says otherwise
cd ~/franka_il/act/franka_lerobot
setsid nohup scripts/queue_cells.sh queues/cotrain_v1 >> queues/cotrain_v1/queue.log 2>&1 < /dev/null &
```
The queue runs the first line of `cells.txt` not in `cells.txt.done`. If that run has `checkpoints/last`, it resumes
(S350_a010 resumes from 20k on CIR2 only). About 10.3 h per run alone on the GPU (0.35 s/step).

## 4. Check on it

```bash
D=~/franka_il/act/franka_lerobot/queues/cotrain_v1
tail -5 $D/queue.log                              # start / end / resume / stop
grep "step:" $D/act_cot_<CELL>_b40.log | tail -1  # step, loss, s/step
cat $D/cells.txt.done                             # finished cells
```
Healthy: startup `realised_sim_fraction` within 0.01 of alpha; about 0.35 s/step alone; loss about 0.05 at 100k.

## 5. Pause and stop (never use `pgrep -f`, it matches its own shell)

```bash
P=$(pgrep -x -f "/bin/bash scripts/queue_cells.sh queues/cotrain_v1"); G=$(ps -o pgid= -p $P | tr -d ' ')
kill -STOP -- -$G   # pause (still holds about 19 GB GPU memory)
kill -CONT -- -$G   # resume
```
To free the GPU:
1. Wait for the next 20k checkpoint, i.e. until `checkpoints/0X0000/training_state/training_step.json` exists.
2. Stop the queue: `kill $P`.
3. Stop training: `kill -- -$G`.

Restarting the queue resumes from that checkpoint.

## 6. When a run finishes

**On CIR1:** push the model to the Hub (private) as `Lusmse/act_franka_cot_<CELL>_b40`, path
`checkpoint_100000/pretrained_model` (same layout as models 8 and 9). CIR2 downloads it and links it for the robot eval.

**On CIR2** (robot eval kit):

```bash
ln -sfn ~/jello/JELLO_ARM_FRANKA/outputs/train/act_cot_<CELL>_b40/checkpoints/100000/pretrained_model \
        ~/act_eval/models/<n>_c<k>_cot_<CELL>
```
Add a row to `~/act_eval/README.md`; serve with `~/act_eval/serve.sh <n>`.
Done: 8 = S350_a090, 9 = S350_a100 (sim-only).

## 7. Cells (priority order = `cells.txt`)

| Tier | Cells |
|---|---|
| done (CIR2) | S350_a090 (model 8, HF `Lusmse/act_franka_cot_S350_a090_b40`), S350_a100 (model 9, HF `Lusmse/act_franka_cot_S350_a100_b40`); real-only R50 = model 4 (batch 32, footnoted) |
| 1: end points | S350_a010 (restarts from 0 on CIR1; CIR2's 20k partial is not used), S050_a090, S050_a100, R25 |
| 2: midpoints | S350_a050, S200_a090, S200_a100 |
| 3: curve shape | S350_a095, S350_a070, S100_a090, S050_a050 |
| 4: fill-in | S350_a030, S350_a080 |
| later | R75, R100 (R150) need new teleop demos (new dataset build: footnote the stats); 2 extra seeds at alpha* |

Names: `S<N_sim>_a<alpha x100>` with 50 real (0 real when alpha = 1.0); `R<N_real>` = real-only.
