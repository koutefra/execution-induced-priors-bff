# CuBFF Experiments: Interaction vs Noise-Interaction

This repository contains a working copy of CuBFF that I used to run
Computational Life BFF experiments and analyze self-replicator dynamics.

The study focuses on two questions:

- Q1 (attribution): Is the interaction regime's speedup in self-replicator
  discovery explained by execution-induced filtering alone, or does it require
  lineage-dependent population feedback?
- Q2 (post-takeover dynamics): After replicators spread, do we see continued
  innovation, or convergence to an evolutionary stasis dominated by minimal
  motifs?

Two experiment modes are used:

- R: interaction (runs/interaction)
- N: noise-interaction with random partner (runs/random)

Runs are 16k epochs each, soup size is 2^17, and mutation probability is 0.00024.

## Dependencies

On Debian-based systems:

  sudo apt install build-essential libbrotli-dev

On Arch Linux:

  pacman -S brotli base-devel

You can also use the provided flake.nix:

  nix develop

## Build

  make

Or build without CUDA:

  make CUDA=0

## Running Experiments

The run scripts in the repo match the setup used for these results.
They write logs to runs/<mode>/:

- log_<seed>.log (CSV log with self-replicator tapes)
- out_<seed>.log (stdout output with glyph frequency snapshots)
- err_<seed>.log (stderr)

Interaction (R) runs (seeds 0..19):

  ./run_interaction.sh

Noise-interaction (N) runs (seeds 0..99):

  ./run_random_interaction.sh

You can also run a single seed directly:

  ./run_seed.sh interaction 0
  ./run_seed.sh random 0

Log format (log_<seed>.log):

  epoch,brotli_size,soup_size,higher_entropy,number_selfreps,selfrep_tapes,selfrep_scores

## Replicator Families

I classify replicators into four families by loop structure and copy mechanism:

1) 9-symbol replicators (reverse-copy core)
2) 10-symbol replicators (9-symbol with an extra bracket)
3) Palindromic replicators (10-symbol subclass with L2 = reverse(L1))
4) Offset replicators (analytically identified; unobserved in runs)

Strong vs weak reverse replicators:

- Strong: head_1 writes to foreign tape, head_0 reads its own tape.
- Weak: head_0 enters foreign tape and writes (susceptible to zero poisoning).

## Analysis Scripts

All analysis scripts live in scripts/.

### Replicator detection and classification

- Find first replicator per log:

  python3 scripts/find_first_replicator.py interaction
  python3 scripts/find_first_replicator.py random

- Report replicators and classify by family:

  python3 scripts/report_replicators.py interaction > /tmp/reps.txt
  python3 scripts/analyze_replicator_types.py < /tmp/reps.txt

### Epoch glyph distributions (top/bottom 16)

- Extract glyph frequencies for each printed epoch:

  python3 scripts/extract_epoch_glyphs.py runs/interaction/out_0.log

- Aggregate overlap and collapse metrics (produces plots and CSV):

  python3 scripts/analyze_epoch_glyphs.py

Outputs:

- analysis_epoch_glyphs.csv
- analysis_plots/overlap_boxplots.png
- analysis_plots/collapse_index.png

### Replicator-level symbol distributions

For takeover runs {1,2,8,10,12,18}, this script:

- aggregates all replicator tapes in epochs 15872-512 .. 15872
- decodes tapes into glyphs
- prints top-16 and bottom-16 symbols
- prints mean higher_entropy over 15360..15672

  python3 scripts/analyze_replicator_glyphs.py

You can override defaults with flags:

  --epoch-end 15872
  --epoch-span 512
  --entropy-start 15360
  --entropy-end 15672
  --runs 1,2,8,10,12,18

## Takeover Runs

Non-takeover runs (R_null):

  {0,3,4,5,6,7,9,11,13,14,15,16,17,19}

Takeover runs (R_post):

  {1,2,8,10,12,18}

Takeover start epochs (for R_pre):

  run 1 -> 4496
  run 2 -> 1158
  run 8 -> 7017
  run 10 -> 13905
  run 12 -> 2191
  run 18 -> 12157

R_pre is computed from the last printed epoch before takeover.
R_post is computed at epoch 15872.

## Metrics

Overlap metric (Top-16 / Bottom-16):

  O(A,B) = |S_A intersect S_B|

Reported as mean +/- std, plus normalized value O/16.

Rank stability:

  Sim(A,B) = sum_s rank_A(s) * rank_B(s)

Collapse index (per run):

  C = M / L

where M is mean frequency of top-16 symbols and L is mean frequency
of bottom-16 symbols.

## Expected Qualitative Outcomes

- R_null and N show higher overlap with themselves than cross-regime overlaps.
- R_pre overlaps strongly with R_null, indicating similar pre-takeover bias.
- R_post overlaps are low and show higher collapse indices, reflecting lineage
  dominance.
- Collapse index for R_post is significantly higher than for R_null and N.

## Repro Checklist

1) Build: make
2) Run R and N experiments (see run_* scripts)
3) Generate analyses:

   python3 scripts/analyze_epoch_glyphs.py
   python3 scripts/analyze_replicator_glyphs.py

