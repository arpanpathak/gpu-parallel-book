# About the cover

The cover is the inside of a hive. The queen bee gives a single order, `c[i] = a[i] + b[i]`, and 32 worker bees carry it out together, each on its own honeycomb cell. That is the picture this book keeps coming back to.

- The **queen** is the host program on the CPU. She does no filling herself. She decides the work and launches it.
- The **order** is a kernel: one piece of code, written once.
- The **workers** are GPU threads. Each runs the same order on a different cell.
- The **32 workers in formation** are one *warp*, the group of 32 threads that a GPU steps through each instruction together.
- The **empty cell** belongs to the worker who wandered off on a branch of its own. When threads in a warp take different branches, the warp has to wait for the stragglers. Chapter 5 calls this *divergence*.
- The **crab** is the Rust crew. From Part V onward, the kernels are written in Rust with cuda-oxide.

Each chapter opens with a member of the cast at work on the chapter's subject.

| Chapter | Character | Scene |
|---|---|---|
| 1 | Tally the ant | a column of ants crossing a meadow all at once, then crossing a one-ant bridge in single file: the serial part of the work |
| 2 | Comb the worker bee | a guide beside a cut-away hive whose comb frames are SMs, with workers standing on them in formations that move as one: warps |
| 3 | The Queen | holds up a launch, <<<4, 256>>>, beside 4 comb frames of worker bees: 1 order, carried out by every thread of every block |
| 4 | Ferry the pelican | carries crates of bytes on a raft from the host memory bank to the device memory bank: pinned data crosses in 1 trip |
| 5 | Whistle the meerkat | keeps watch at a gate that stays down until every bee has arrived: a barrier |
| 6 | The otter twins | float down 2 rivers side by side, each carrying work: 2 streams at the same time |
| 7 | Inch the caterpillar | walks with each segment over the next numbered cell: neighbouring threads read neighbouring addresses |
| 13 | Pinch the crab | one claw for `a[i]`, one for `b[i]`, the sum dropped into `c[i]`; two spare crabs doze past the end of the row |

The plates are drawn by `tools/art.py`.
