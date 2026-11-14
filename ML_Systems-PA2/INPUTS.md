# Models, requests, and targets

One run uses one model and one fixed cluster. Your `build_plan` receives the
model key. Every replica in that run serves that model. The runner calls the
same submitted file separately for each case.

For a simple view of the actual supplied requests, open
`public/requests/medium.csv` or the CSV with another case name. It lists arrival
time, prompt length, MAX, and category. The category column is for offline
reading; it is not an extra field in the router API.

The JSON files under `public/cases` are complete simulator inputs. They also
contain internal configuration and realized outputs for reproducibility.
Your policy receives only the fields in `tp_api.py`. It must not read a trace,
future arrivals, or look up an answer using a request ID.

## Six case names

| Case | Model key | Input pattern |
|---|---|---|
| small | `8b` | Steady arrivals with a mix of prompt lengths |
| small-shift | `8b` | Chat, a surge of long prompts, then recovery |
| medium | `70b` | Steady arrivals with a mix of prompt lengths |
| medium-shift | `70b` | Chat, a surge of long prompts, then recovery |
| large | `400b` | Steady arrivals with a mix of prompt lengths |
| large-shift | `400b` | Chat, a surge of long prompts, then recovery |

The public files contain specific requests. A new grading seed creates new
requests under the same rules below. It does not add a new model, GPU type,
request category, or network condition. Both traffic patterns have equal weight
within each model; the overall score also gives extra weight to the lowest model.

## Three short practice cases

`small-practice` and `medium-practice` use the first eight requests
(q0000–q0007) of the corresponding sustained case. `large-practice` uses
q0000, q0001, q0006 and q0007 from `large`. Each selected request retains its original arrival time,
prompt length, MAX, actual output, and latency targets. The cases contain
chat, RAG and long-context requests, including same-time and later arrivals.
Only the request list and case ID differ from the original case. They are
not a representative traffic sample: the request mix and load are much
shorter and do not test the changing-load case. Practice scores are not
the final score. `python run.py --all` and `--check --all` keep their
original meaning and use only the six regular cases. Run a practice case
individually, e.g. `python run.py --case large-practice`.

## The three request categories

These are the only categories in this edition. Every listed prompt length and
MAX value can occur for that category. They describe lengths, not natural
language content. RAG means Retrieval-Augmented Generation.

| Category | Possible prompt lengths | Possible MAX output lengths |
|---|---|---|
| Chat | 256, 512, 1,024 | 64, 128 |
| RAG | 2,048, 4,096, 8,192 | 128, 256 |
| Long context | 16,384, 32,768 | 256, 512 |

The router sees the prompt length and MAX. It does not receive a category
field. Each category has a distinct prompt-length range in this assignment.

Actual output can stop before MAX. On `small` and `medium` inputs, actual
outputs are 16–48 tokens for chat, 32–96 for RAG, and 64–192 for long context.
On `large` inputs, the ranges are 16–32, 24–48, and 32–64. Within each range,
the generator samples an integer uniformly. These shorter large-model outputs
limit CPU simulation cost. They do not speed up the simulated GPU clock.
The policy learns generated progress, never the future stopping length.

The wrapper reserves memory for **prompt + MAX**, then releases that request's
reservation when it ends. This differs from PagedAttention: admission does not
allocate extra KV blocks one token at a time. There is no eviction, swapping,
or request migration.

## Arrival patterns

Small and medium inputs contain 60–84 requests. Large inputs contain 36–48.
They arrive over time, including groups of 1, 2, or 4 at the same timestamp.
The wrapper routes those requests one by one. Each admission updates memory
and resident counts before the next route call.

In steady traffic, the average mix is 65% chat, 25% RAG, and 10% long context.
In a shift input, the three successive portions have these average mixes:

| Portion | Chat | RAG | Long context | Typical load |
|---|---:|---:|---:|---|
| Start | 80% | 15% | 5% | Lower |
| Surge | 20% | 35% | 45% | Higher |
| Recovery | 60% | 25% | 15% | Moderate |

These are averages, not exact counts. Random sampling changes the mix and
spacing between arrivals. Your code receives current requests and state,
not a forecast or a signal announcing each portion. The actual public input
files show examples of the variation. Exact sampling parameters are available
in `public/distribution.json` for reference; no probability model is required
in your answer.

## Latency targets

TTFT is time from arrival to first output token. ITL is the gap between two
successive output tokens. Each request passes only if its TTFT and its own
95th-percentile ITL both meet the targets. All values below are milliseconds.

| Model | Chat TTFT / ITL | RAG TTFT / ITL | Long-context TTFT / ITL |
|---|---|---|---|
| 8b | 1,000 / 50 | 3,000 / 100 | 12,000 / 300 |
| 70b | 2,000 / 120 | 8,000 / 250 | 30,000 / 800 |
| 400b | 32,000 / 1,400 | 140,000 / 2,800 | 480,000 / 8,000 |

The result page shows each request's measured value beside its target.
You do not set these targets in the code.

Each request earns **0 points** if it misses either target. Otherwise it earns:

- 70 points for meeting both targets.
- Up to 20 points for latency margin: `20 * (1 - max(TTFT/TTFT_target, P95_ITL/ITL_target))`.
- Up to 10 points for drain efficiency: `10 * E`.

`E` is an optimistic drain-time lower bound divided by measured drain time.
Drain is the time from the last arrival to the last completion. The bound uses
the offered work and fixed hardware, not a teacher policy; it is not a claim
that 100 points is achievable. Within one input, average the request points
in each type, then average the types present. Every request must complete.

For the overall score, first combine requests of the same type across the
preselected seeds **within the same model and traffic pattern**. Average the
type scores, then average the two traffic patterns for each model. Finally:

`overall = 0.75 * average of the three model scores + 0.25 * lowest model score`.

The public `--all` run uses the same rule with the one provided public input
per condition. The single-input score is useful for trial and error; it is not
the overall score. This automatic performance score is not the assignment grade. See README.md for the 60 base + 40 rank grading rule.

## Fixed hardware

There are 32 identical teaching GPUs, with 8 per node. GPU IDs 0–7 belong to
node 0, 8–15 to node 1, 16–23 to node 2, and 24–31 to node 3.

| Quantity | Teaching constant |
|---|---|
| Memory per GPU | 80 GB |
| Compute per GPU | 150 trillion floating-point operations per second |
| High-bandwidth memory transfer per GPU | 1,500 GB/s |
| Link bandwidth within a node | 200 GB/s |
| Link bandwidth between nodes | 25 GB/s |
| Link latency within / between nodes | 500 / 5,000 ns |

GB uses decimal bytes. These constants do not describe a measured commercial
GPU. GPUs cannot be rented, added, or removed. Leaving a GPU unused does not
change the resource budget.

| Model | Total weight memory | TP widths that can fit model weights |
|---|---:|---|
| 8b | 16 GB | 1, 2, 4, 8 |
| 70b | 140 GB | 2, 4, 8 |
| 400b | 800 GB | 16 |

For example, although the 400b model profile describes TP1, TP2, TP4 and TP8
as tensor layouts, their per-GPU weight memory exceeds the available GPU
memory. They are not valid replica choices in this assignment. Weight fit is
necessary but not sufficient for admitting a request: the KV reservation and
buffers also need memory.

The shared validator also checks buffers and MAX KV reservations. You do not
have to calculate admission memory yourself. TP32 is not supported by these
model profiles. The 400b layout has fewer choices; 70b is the main example for
comparing different TP widths.

The wrapper fixes execution grouping and scheduling. Pipeline parallelism,
prefill/decode separation, and batch-size tuning are outside the student
interface. The policy can change the GPU groups and request destinations.

The wrapper automatically batches **up to 8 ready requests of the same
phase** on each replica. It does not wait for a batch to fill. Prompt work
and decode work are separate groups, chosen by oldest ready work. A request
can join a later group when its previous step finishes; it leaves at EOS.
Students do not set a batch size.

Requests in one group share model-weight reads. The teaching compute model
pads a group to its longest prompt (or current context), rounded up to 128
tokens. Mixing very different lengths can therefore waste work. Your routing
choices affect which requests can be batched together. MAX KV admission is
still based on each request's exact prompt and output cap; batching does not
change that reservation rule. This is a teaching batching model, not a
faithful implementation of a production serving engine.

The 2026-10-26 edition uses automatic batches up to 8. Measurements from the
2026-10-25 edition used a group limit of 1 and are not directly comparable.

Sources: `tp_api.py`, `tp_bridge.py`, `public/distribution.json`,
`public/cases/*.json`, `_engine/simulator/api.py`, and `_engine/lab_score.py`.
