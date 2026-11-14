# PA2: GPU groups and request routing

Use Python 3.10 or newer on Linux, for example Ubuntu 22.04 LTS or newer.
Windows users may use Ubuntu in WSL. No physical GPU is required. See
[SETUP.md](SETUP.md) for installation. The editable slide guide is provided
separately by the course.

In PA1, M1 was a basic implementation and M2 was a competition. In PA2 there
is one assignment: improve your placement and routing policy for the
competition. There is no separate M1 implementation to submit.

## What you write

Edit **student.py**. It already contains Answer A from the separately supplied
slides, so you can run it immediately. To try Answer B, copy its code from
the slides into `student.py`, save the file, and run `python run.py` again.
The file contains two functions:

- `build_plan(spec)` returns GPU groups (`Replica` objects). Their positions in
  the returned list are replica IDs: 0, 1, 2, and so on.
- `route(request, state)` chooses the ID of a candidate replica for each
  request. The state describes the current candidates and their load.

The course provides the workloads and a wrapper that validates your answer,
manages request state, and calls the **ASTRA-sim simulator** in the background.
ASTRA-sim executes the generated compute/communication workload on its model
of the GPU network and returns simulated timings. It does not generate text.
You do not write CUDA, workload JSON, or an execution graph.

## Set up and run

After extracting the ZIP, open a Linux terminal in the folder containing
`student.py` and `run.py`. Install ASTRA once, following [SETUP.md](SETUP.md):

```bash
python setup.py
```

Then run your current `student.py`:

```bash
python run.py
```

This runs the default **medium** case with ASTRA and writes the output to
`results/latest/index.html`. Open that file in a browser to see the score,
GPU groups and request latencies. Each execution also saves a separate result
folder with a copy of the code used. After editing `student.py`, run the same
command again to see the new result.

You can check the layout and one routing decision for each public case before
running ASTRA:

```bash
python run.py --check --all
```

The check does not exercise every future routing decision. A full run checks
each decision as requests arrive and reports errors with a repair hint.

## Try a shorter input first

The small and medium practice cases use the first 8 requests of the corresponding
sustained inputs. Large practice uses requests q0000, q0001, q0006 and q0007
from the large sustained input. Use the same `student.py` and native ASTRA simulator:

```bash
python run.py --case small-practice
python run.py --case medium-practice
python run.py --case large-practice
```

They are useful for a quick smoke test, but are too short to represent the
full traffic, overload, or the final score. Test the six regular cases
before submission. `--all` runs only those six regular cases. Runtime
depends on your CPU, other jobs on the computer, and your policy.

## How long will a run take?

The following are measured examples, not time limits. On 2026-10-31, we ran the supplied
`student.py` once per case on Ubuntu 24.04.3, Python 3.12.3, and an Intel
Core Ultra 9 285K (24 CPU cores), with ASTRA already installed. Values are
wall-clock time from launching the command to its completion. Your CPU,
other users/jobs, and policy can change the time substantially.

| Case | Example elapsed time |
|---|---:|
| `small-practice` | about 0.6 s |
| `medium-practice` | about 2.8 s |
| `large-practice` | about 28.8 s |
| `small` | about 3.8 s |
| `small-shift` | about 4.2 s |
| `medium` | about 22.8 s |
| `medium-shift` | about 17.6 s |
| `large` | about 209.6 s |
| `large-shift` | about 77.6 s |

`--all` runs the six regular cases sequentially. Setup time is not included.


## Assignment grade: 60 base points + 40 rank points

Your local result is a **performance score**, not your final assignment grade.
The course evaluates every submission on the same grading cases. A submission
is valid if `student.py` is present and all grading cases complete with a valid
plan and valid routing decisions. Some requests may miss their latency targets;
those misses reduce the performance score but do not make the submission invalid.
The supplied starter answer is eligible if it completes all grading cases.

Each valid submission earns 60 base points. The remaining 5–40 points depend on
its rank by the unrounded overall performance score, as in PA1. Let N be the
number of valid submissions and r be 1 plus the number of valid students with a
strictly higher unrounded performance score:

`group = 1 + floor(8 * (r - 1) / N)`
`competition points = 45 - 5 * group`
`assignment grade = 60 + competition points`

The eight groups receive 40, 35, 30, 25, 20, 15, 10, and 5 points.
Exactly equal performance scores receive the same rank and the same grade; the next rank skips
the positions occupied by the tie. If N is 1, that valid student receives 100.
If all valid students tie, all receive 100. A missing or invalid submission earns
0 and is excluded from the ranking. Infrastructure failures are reviewed rather
than counted as student failures; the ranking remains pending until resolved.

When N is less than 8, some groups are empty. For example, five valid students
without ties have grades 100, 95, 85, 80, and 70.
See the separate slide guide for examples and details.

## Inputs

A **case** is one input: one model and one traffic trace. A **run** evaluates
your answer on one case. `--all` runs the six cases one after another.

| Case | Model | Traffic |
|---|---|---|
| `small` | 8B | Sustained |
| `small-shift` | 8B | Changing load |
| `medium` | 70B | Sustained; default |
| `medium-shift` | 70B | Changing load |
| `large` | 400B | Sustained |
| `large-shift` | 400B | Changing load |

```bash
python run.py --list
python run.py --case medium-shift
python run.py --all
```

One answer must handle all three models and both traffic patterns. The case
names `small`, `medium`, and `large` describe the **model size**, not a
particular request's length. Request arrival times, prompt lengths and maximum
output lengths for each case can be inspected in [public/requests](public/requests).
The model and traffic rules, latency targets and scoring are in [INPUTS.md](INPUTS.md).
The API fields are defined in [tp_api.py](tp_api.py).

The wrapper reserves KV memory for the prompt plus maximum output length until
the request ends. Unlike PagedAttention, this assignment does not allocate KV
cache one block at a time. Actual output lengths and future arrivals are not
shown to your code. Ready requests in the same phase can be batched automatically;
you do not choose the batch size.

Submit `student.py` through the course portal. Follow the course announcement
for the deadline and any separate report requirement.
