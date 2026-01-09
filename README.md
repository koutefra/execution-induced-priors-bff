# Execution-Induced Priors in BFF

Modified CuBFF implementation for the paper "Execution-Induced Priors Explain Replicator Discovery in BFF".

We run controlled experiments to understand what drives self-replicator discovery in the BFF system:
- Does it require population-level selection, or is execution-induced filtering enough?
- After takeover, do replicators keep evolving or just stabilize?

**Regimes:**
- `R` (interaction): Standard primordial soup with pairwise interactions
- `N` (noise-interaction): Each program interacts with fresh random noise
- `U` (uniform): Pure random sampling baseline

Default config: 16k epochs, soup size 2<sup>17</sup>, mutation rate 0.024%

## Build

```bash
make
```

Without CUDA:
```bash
make CUDA=0
```

**Dependencies** (Debian/Ubuntu):
```bash
sudo apt install build-essential libbrotli-dev
```

Arch:
```bash
pacman -S brotli base-devel
```

Or use the nix flake:
```bash
nix develop
```

## Running Experiments

All run scripts write to `runs/<mode>/`:
- `log_<seed>.log` - CSV with replicator tapes
- `out_<seed>.log` - stdout with glyph frequencies
- `err_<seed>.log` - stderr

Run full experiments:
```bash
./run_interaction.sh      # R regime (seeds 0-99)
./run_random_interaction.sh  # N regime (seeds 0-99)
./run_reinit.sh      # U regime (seeds 0-99)
```

Single runs:
```bash
./run_seed.sh interaction 0
./run_seed.sh random 0
./run_seed.sh reinit 0
```

## Analysis

All scripts in `scripts/`.

**Discovery times:**
```bash
python3 scripts/find_first_replicator.py [regime]
```

**Replicator classification:**
```bash
python3 scripts/report_replicators.py interaction > /tmp/reps.txt
python3 scripts/analyze_replicator_types.py < /tmp/reps.txt
```

**Symbol distributions:**
```bash
python3 scripts/analyze_epoch_glyphs.py
python3 scripts/analyze_replicator_glyphs.py
```

Outputs plots to `analysis_plots/` and stats to `analysis_epoch_glyphs.csv`.