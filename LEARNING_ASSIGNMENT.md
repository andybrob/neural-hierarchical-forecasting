# Astraeus: Learning Assignment

**Research question:** Can hierarchical multi-scale decomposition + cross-variate attention + adaptive seasonal priors outperform models with static seasonal components on real commercial data (M5 / Walmart)?

**Your starting point:** Strong Python and C. No C++ experience. No CUDA experience. Some PyTorch.

**How this works — the loop for every task:**

```
TUTORIAL → ATTEMPT → FEEDBACK
```

1. **Tutorial** — consume the listed resource(s) before writing any code. Required, not optional.
2. **Attempt** — close the tutorial and build it from memory. Fill in the attempt log.
3. **Feedback** — paste your attempt log to Claude with: *"I'm on Task X.Y of Astraeus. Here is my attempt."*
   Claude will ask questions to probe understanding, give hints, and only provide a reference
   implementation after you have demonstrated that you understand *why* your attempt failed.

**Status markers:** `[ ]` not started · `[~]` in progress · `[x]` reviewed and understood

---

## Bucket 0: CUDA From Zero

*You have C experience, which means you already understand pointers, memory layout, and manual memory management. CUDA is a dialect of C, not C++. The core mental model shift is: instead of one thread doing everything sequentially, you launch thousands of threads that each do a tiny piece.*

*Work through these in order. Each task builds on the one before. Do not skip ahead.*

---

### Task 0.0 — CUDA Mental Model and First Kernel

**What:** Understand the CUDA thread hierarchy and write your first kernel: vector addition.

**Tutorial (do this first):**
- [CUDA by Example, Ch. 1–4](https://developer.nvidia.com/cuda-example) — free PDF. Ch. 1-2 is context, Ch. 3-4 is your first kernels. ~2 hours.
- [GPU MODE Lecture 1](https://www.youtube.com/watch?v=LuhJEEJQgUM) — 45 min. Covers grids, blocks, threads with PyTorch context.

**Attempt:** Without re-reading the tutorial, write a CUDA kernel that adds two float arrays element-wise. Then a second kernel that squares every element. Compile and run both.

**Key concepts to internalize before moving on:**
- `gridDim`, `blockDim`, `blockIdx`, `threadIdx` — what each refers to
- How to compute a global thread index: `int i = blockIdx.x * blockDim.x + threadIdx.x`
- Why you need a bounds check (`if (i < n)`)
- The difference between host memory (`malloc`) and device memory (`cudaMalloc`)
- What `cudaMemcpy` directions mean (`cudaMemcpyHostToDevice` vs `cudaMemcpyDeviceToHost`)

**Attempt log:**
```
Date:
What I built:
What confused me:
What the bounds check is protecting against:
```

---

### Task 0.1 — Parallel Reduction (RMSSE Kernel)

**What:** Write a CUDA kernel that computes the RMSSE metric across a batch of predictions on the GPU. This is the loss function used later for training.

**Formula:**
```
denominator_i = mean((y_{i,t} - y_{i,t-1})^2)  # naive forecast error per series, pre-computed once
RMSSE = sqrt( mean((y - y_hat)^2) / denominator )
```

Parallel reduction is the canonical "learn CUDA" exercise because it forces you to confront shared memory, warp divergence, and why naive parallelism is slower than sequential for small inputs.

**Tutorial (do this first):**
- [Mark Harris, Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) — NVIDIA PDF. Read all 7 optimization steps. ~3 hours.
- [GPU MODE Lecture 9](https://christianjmills.com/posts/cuda-mode-notes/lecture-009/) — ML angle on reductions.

**Attempt:** Implement RMSSE in three stages, measuring performance at each:
1. Naive: one thread per element writes squared errors to a buffer, then `thrust::reduce` finishes it
2. Shared memory reduction: load a tile into shared memory, reduce within the tile
3. Warp shuffle reduction: replace shared memory with `__shfl_down_sync`

**Key concepts to internalize:**
- Why is the naive version memory-bandwidth bound?
- What is a warp (32 threads), and why do warp shuffles not need shared memory?
- What are shared memory bank conflicts and how do you avoid them?
- Why does a parallel reduction of N elements only need log2(N) steps?

**Attempt log:**
```
Date:
Which stages I completed:
Performance numbers (naive vs. shared vs. warp shuffle):
What bank conflicts are and whether I avoided them:
What __shfl_down_sync is doing at the warp level:
```

---

### Task 0.2 — Fused LayerNorm Kernel

**What:** Implement a LayerNorm kernel that computes mean, variance, and normalization in a single pass using Welford's algorithm. This teaches kernel fusion — combining operations that would normally be separate kernel launches into one.

**Why this matters:** In a Transformer, LayerNorm runs after every attention and FFN layer. Fusing it eliminates extra global memory reads/writes. This is also a building block for the attention kernel later.

**Tutorial (do this first):**
- [OneFlow LayerNorm post](https://oneflow2020.medium.com/how-to-implement-an-efficient-layernorm-cuda-kernel-oneflow-performance-optimization-731e91a285b8) — read in full before coding. ~1.5 hours.
- Welford's algorithm: look it up separately first and derive it on paper. Understand the update rule before touching CUDA.

**Attempt:**
1. Two-pass: one kernel for mean+variance (reduction), one kernel to normalize. Get it correct first.
2. Fused single-pass using Welford's online algorithm.
3. Verify both against `torch.nn.functional.layer_norm` (max absolute error < 1e-5).

**Key concepts to internalize:**
- Why does Welford's algorithm give numerically stable variance without a second pass?
- Where does the memory traffic come from in the two-pass version that fusion eliminates?
- What is occupancy and how does shared memory usage affect it?

**Attempt log:**
```
Date:
Two-pass implementation: correct? (yes/no, max error vs. PyTorch):
Fused implementation: correct? (yes/no, max error vs. PyTorch):
Speedup of fused over two-pass:
Welford update rule in your own words:
```

---

### Task 0.3 — Tiled Matrix Multiplication

**What:** Write a tiled GEMM (general matrix multiply) kernel. This is the prerequisite for FlashAttention. The tiling argument in FlashAttention is a direct generalization of this.

**Why this matters:** Every linear layer in a Transformer is a matrix multiply. FlashAttention tiles the QK^T computation the same way. If you don't understand tiled GEMM, FlashAttention will be opaque.

**Tutorial (do this first):**
- [Simon Boehm's SGEMM tutorial](https://siboehm.com/articles/22/CUDA-MMM) — the best single resource for this. Work through all steps. ~4 hours.
- Read Ch. 4-5 of Programming Massively Parallel Processors if you want the textbook version.

**Attempt:**
1. Naive: one thread computes one output element (C[i][j] = dot product of row i of A and col j of B). Correct but slow.
2. Tiled: load tiles from A and B into shared memory, compute the tile product, accumulate. The classic optimization.
3. Profile: measure FLOPS utilization vs. cuBLAS. Aim for >50% of cuBLAS.

**Key concepts to internalize:**
- Why does the naive kernel reload data from global memory O(N) times more than necessary?
- Where do you need `__syncthreads()` and why exactly?
- What is register pressure and how does it limit tile size?
- Why does (128, 128) block tiling specifically appear so often?

**Attempt log:**
```
Date:
Naive kernel: correct? Performance vs. cuBLAS (%):
Tiled kernel: correct? Performance vs. cuBLAS (%):
Where __syncthreads() is needed and why (in your own words):
What "register pressure" means:
```

---

### Task 0.4 — Fused Row-Wise Softmax

**What:** Implement a numerically stable, row-wise softmax using the online max+sum algorithm. This is the exact building block inside FlashAttention.

**Why this matters:** Standard softmax requires reading the input twice: once to compute the max (for numerical stability), once to compute exp and normalize. FlashAttention fuses this into one pass. Understanding how is the key insight of the paper.

**Tutorial (do this first):**
- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867) — short paper, read in full. ~30 min.
- Derive the correction formula on paper before coding: if your running max increases from m_old to m_new, how do you correct previously accumulated exp values?

**Attempt:**
1. Naive three-pass: max pass → exp+sum pass → divide pass.
2. Online two-pass: running max + corrected sum in one pass, then divide.
3. Verify both are numerically identical on inputs that don't cause overflow in the naive version.

**Key concepts to internalize:**
- What is the correction factor when the running max is updated?
- Why does this matter for FlashAttention (what does it allow you to avoid storing)?

**Attempt log:**
```
Date:
Naive softmax: correct?
Online softmax: correct?
The correction formula in your own words (derive it here, not just cite it):
What this enables in FlashAttention:
```

---

### Bucket 0 Checkpoint

Before starting Bucket 1, answer these four questions without looking anything up:

1. Why is a naive parallel reduction slower than a sequential CPU scan for small arrays (<1024 elements)?
2. What is a warp, and why does `__shfl_down_sync` not require shared memory?
3. In tiled GEMM, why do you need `__syncthreads()` at exactly two points per tile iteration?
4. FlashAttention tiles the QK^T multiply. What does it keep in SRAM, and what does this avoid materializing?

```
Answers:
1.
2.
3.
4.
```

---

## Bucket 1: Architecture Core

*Build the model components in PyTorch (Python). Use `torch.utils.cpp_extension.load_inline` to call your CUDA kernels from Bucket 0 where relevant — this avoids the pain of a full C++ build system while still giving you the integration experience.*

*Note on C++: LibTorch uses C++ conventions (templates, namespaces, references). You will encounter C++ syntax when writing kernel wrappers. The rule: keep the C++ wrapper minimal — just enough to expose the kernel to Python. All kernel logic stays in `.cu` files written in C-style CUDA, which you already understand.*

---

### Task 1.1 — N-HITS Hierarchical Blocks

**What:** Implement the N-HITS residual block stack in PyTorch. The key idea: different blocks downsample the input to different resolutions before applying their MLPs, so each block specializes in a different frequency band.

**Tutorial (do this first):**
- Read the [N-HITS paper](https://arxiv.org/abs/2201.12886). Focus on Section 3 (architecture). Sketch the block diagram on paper before reading the code section.
- Before writing code, answer on paper:
  - What is a "backcast"? What is a "forecast"?
  - How does subtracting the backcast from the input at each block enable specialization?
  - Why does downsampling before the MLP help for long-horizon forecasting?

**Attempt:**
1. Implement one block: downsample → MLP → upsample → split into backcast + forecast.
2. Stack multiple blocks with residual subtraction.
3. Test on a synthetic multi-frequency signal (e.g., 7-day + 365-day seasonality). Verify the blocks decompose it.

**Attempt log:**
```
Date:
Backcast vs. forecast in your own words:
Why residual subtraction enables frequency specialization:
Test result on synthetic signal (did blocks decompose it?):
```

---

### Task 1.2 — Cross-Variate Attention

**What:** Implement attention that operates *across series* (the iTransformer inversion), not across time steps.

**The inversion:** Standard attention: N series, T time steps → T tokens of dimension N. iTransformer: T tokens of dimension N → N tokens of dimension T. Each series is one token; attention learns cross-series relationships.

**Tutorial (do this first):**
- Read the [iTransformer paper](https://arxiv.org/abs/2310.06625), Section 3.2 on the inversion specifically.
- Before coding: draw the QKV tensor shapes for both standard and inverted attention on paper.

**Attempt:**
1. Standard time-axis attention (warm-up). Verify against `torch.nn.MultiheadAttention`.
2. Inverted (cross-variate) attention. Verify shapes are correct.
3. Replace the softmax inside your attention with your Bucket 0 fused softmax kernel via `load_inline`.

**Attempt log:**
```
Date:
Q, K, V shapes for 50 series × 168 time steps (standard vs. inverted):
What the attention weights represent in the inverted case:
Did the fused softmax kernel integration work?
```

---

### Task 1.3 — FlashAttention Forward Kernel

**What:** Implement the FlashAttention forward pass as a CUDA kernel. The hardest task in the project. Budget significant time.

**Prerequisites:** Tasks 0.3 and 0.4 must be complete and understood. Seriously.

**Tutorial (do this first):**
- [lubits.ch FlashAttention series, Part 1](https://lubits.ch/flash/Part-1) — builds up CUDA building blocks before the full kernel.
- [GPU MODE Lecture 12](https://christianjmills.com/posts/cuda-mode-notes/lecture-012/) — conceptual walkthrough.
- Implement in **Triton first** using [this walkthrough](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/). Triton lets you focus on the algorithm without fighting CUDA syntax. Get it correct. Then port.

**Attempt:**
1. Triton implementation — correct vs. `torch.nn.functional.scaled_dot_product_attention`.
2. CUDA port — correct (same numerical check).
3. CUDA version faster than naive attention at sequence length ≥ 256.

**Attempt log:**
```
Date:
Triton implementation: correct?
CUDA implementation: correct?
CUDA speedup over naive (at what sequence length):
FlashAttention tiling strategy in your own words (before opening the paper):
```

---

### Bucket 1 Checkpoint

1. N-HITS: if you have 3 stacked blocks with downsampling factors [8, 4, 1], what is each block "responsible for" intuitively?
2. iTransformer: for 50 series × 168 time steps, write out the Q, K, V shapes in the inverted formulation.
3. FlashAttention: what is the memory complexity of standard attention vs. FlashAttention in the sequence length N? Why?

```
Answers:
1.
2.
3.
```

---

## Bucket 2: The Novel Contribution

*This is the research work. Buckets 0 and 1 are prerequisites. This is where the project becomes original.*

---

### Task 2.1 — Time-Varying Fourier Coefficients

**What:** Seasonal components whose amplitude and phase drift slowly over time. Motivation: if BFCM spending patterns shift year-over-year, a static seasonal component will be systematically biased.

**Tutorial (do this first):**
- Read the [FITS paper](https://arxiv.org/abs/2307.03756) — shows how far you can get with frequency interpolation alone. Understand what static Fourier features look like in practice.
- Then read the [Fourier time-varying grey model paper (IJF 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0169207023001383) for the time-varying angle.
- Before coding: sketch on paper how you would parameterize amplitude as a slow function of time. What are your design choices?

**Attempt:**
1. Static Fourier baseline: sin/cos at fixed frequencies, learned fixed weights. Verify it helps on seasonal data.
2. Time-varying weights: amplitude/phase are a function of time (your design choice — try linear drift first, then a small state-space model).
3. Regularization: add a smoothness penalty so weights don't overfit noise.
4. Synthetic test: generate a signal with amplitude decaying 5% per year. Fit both static and time-varying models. Does your model recover the decay?

**Attempt log:**
```
Date:
How I parameterized the time-varying weights:
Regularization approach:
Synthetic test result (did the model recover the ~5%/year decay?):
What a static Fourier model would predict vs. what yours predicts 3 years out:
```

---

### Task 2.2 — Concept Drift Validation on Real Data

**What:** Before claiming time-varying seasonality is useful, verify empirically that it actually exists in the data.

**Attempt:**
1. Extract BFCM/holiday indicators from M5 or your own data.
2. Fit a model with static holiday coefficients. Plot those coefficients over time using a rolling window.
3. Run a Chow test or CUSUM test for structural breaks in the holiday coefficients.
4. Document the result honestly: if drift is absent or small, say so. A negative result is still a result.

**Attempt log:**
```
Date:
Dataset used:
Holiday coefficient stability (stable / drifting / unclear):
Structural break test result (p-value, break location if found):
How this changes (or confirms) the Task 2.1 hypothesis:
```

---

### Task 2.3 — Soft Decision Tree Gating (Optional / Exploratory)

**What:** A differentiable decision tree as a gating layer that routes inputs to different model components. The specific design: an oblivious tree (all nodes at depth d use the same split feature) with sigmoid splits, outputting mixing weights for the Fourier block and attention block.

**This is speculative.** It may not beat a simple MLP gate. Run the ablation honestly.

**Tutorial (do this first):**
- Read the [NODE paper](https://arxiv.org/abs/1909.06312), Section 3. Understand what "oblivious" means and why it matters for GPU parallelism.

**Attempt:**
1. Depth-2 oblivious tree with sigmoid splits and temperature annealing.
2. Use it as a gate: output weights that mix Fourier block and attention block.
3. Ablation: tree gate vs. 2-layer MLP gate on the same task.

**Attempt log:**
```
Date:
What "oblivious" means and why it helps with parallelism:
Tree gate vs. MLP gate result:
Decision: keep or drop? Why?
```

---

## Bucket 3: Training and Evaluation

---

### Task 3.1 — End-to-End Training Loop

**What:** Wire all components together and train on M5 (Walmart).

**Attempt:**
1. Connect: N-HITS blocks → cross-variate attention → time-varying Fourier head.
2. Use RMSSE (your Bucket 0 kernel) as the training loss.
3. Add gradient clipping and cosine decay with warmup.
4. Log per-series error distribution, not just aggregate metrics. Tail series matter.
5. Profile the training loop. Identify the slowest component.

**Attempt log:**
```
Date:
Architecture wiring (describe the data flow):
Training curve behavior (stable / NaN / overfit?):
Slowest component in the loop:
Anything surprising in the per-series error distribution:
```

---

### Task 3.2 — Baselines and Ablations

**What:** You cannot claim anything without honest comparison.

**Required runs:**

| Model | Description |
|---|---|
| Seasonal Naive | Predict last observed season |
| N-HITS baseline | Original N-HITS, static Fourier |
| + Cross-variate attention | Add iTransformer-style attention |
| + Time-varying Fourier | Add adaptive seasonality |
| Full model | All components |
| Ablation: no time-varying | Isolate that contribution specifically |
| Ablation: no cross-variate attn | Isolate attention contribution |

**Attempt log:**
```
Results table:
| Model                    | RMSSE | MASE | Notes |
|--------------------------|-------|------|-------|
| Seasonal Naive           |       |      |       |
| N-HITS baseline          |       |      |       |
| + Cross-variate attn     |       |      |       |
| + Time-varying Fourier   |       |      |       |
| Full model               |       |      |       |

Honest conclusion:
What worked, what didn't, what surprised you:
```

---

## Progress Tracker

| Task | Status | Date | Key Insight |
|---|---|---|---|
| 0.0 First kernel | `[ ]` | | |
| 0.1 RMSSE reduction | `[ ]` | | |
| 0.2 Fused LayerNorm | `[ ]` | | |
| 0.3 Tiled GEMM | `[ ]` | | |
| 0.4 Fused softmax | `[ ]` | | |
| Bucket 0 checkpoint | `[ ]` | | |
| 1.1 N-HITS blocks | `[ ]` | | |
| 1.2 Cross-variate attention | `[ ]` | | |
| 1.3 FlashAttention kernel | `[ ]` | | |
| Bucket 1 checkpoint | `[ ]` | | |
| 2.1 Time-varying Fourier | `[ ]` | | |
| 2.2 Concept drift validation | `[ ]` | | |
| 2.3 Soft gating (optional) | `[ ]` | | |
| 3.1 Training loop | `[ ]` | | |
| 3.2 Baselines + ablations | `[ ]` | | |

---

## How to Use This File with Claude

When you complete an attempt log, start a Claude session with:

> *"I'm on Task X.Y of the Astraeus assignment. Here is my attempt log: [paste log]"*

Claude will:
1. Ask questions to probe understanding before giving feedback
2. Give targeted hints before revealing answers
3. Only provide a reference implementation after you demonstrate understanding of *why* your attempt failed
4. Update the key insight column with you once the task is resolved

The contract: you get questions and hints first. Full solutions only after you've shown what you understand.
