#!/usr/bin/env python3
"""
Analyze epoch glyph statistics across interaction (R) and random (N) runs.

This script:
- runs extract_epoch_glyphs.py on each log file
- parses epoch 15872 only
- computes Jaccard overlaps, rank stability, and collapse index
- writes a per-run table and optional plots
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TARGET_EPOCH = 15872
R_NON_TAKEOVER = {0, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17, 19}
R_TAKEOVER_EPOCHS = {
    1: 4496,
    2: 1158,
    8: 7017,
    10: 13905,
    12: 2191,
    18: 12157,
}
TOP_N = 16
BOTTOM_N = 16
PAIR_RE = re.compile(r"(\S+)\(\d+\)=([0-9]*\.?[0-9]+)%")

MAX_RANK_SCORE = 2 * sum(i * i for i in range(1, TOP_N + 1))


@dataclass
class RunData:
    run_id: str
    mode: str
    takeover_status: str
    top_symbols: list[str]
    bottom_symbols: list[str]
    top_freqs: list[float]
    bottom_freqs: list[float]
    ranks: dict[str, int]
    collapse_index: float
    run_index: int


def parse_pairs(line: str) -> tuple[list[str], list[float]]:
    pairs = [(m.group(1), float(m.group(2))) for m in PAIR_RE.finditer(line)]
    symbols = [sym for sym, _ in pairs]
    freqs = [freq for _, freq in pairs]
    return symbols, freqs


def parse_all_epochs(
    text: str,
) -> dict[int, tuple[list[str], list[float], list[str], list[float]]]:
    epochs: dict[int, tuple[list[str], list[float], list[str], list[float]]] = {}
    epoch = None
    most_symbols = None
    most_freqs = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Epoch "):
            try:
                epoch = int(line.split()[1])
            except (IndexError, ValueError):
                epoch = None
            most_symbols = None
            most_freqs = None
            continue
        if epoch is None:
            continue
        if line.startswith("most:"):
            most_symbols, most_freqs = parse_pairs(line)
            continue
        if line.startswith("least:") and most_symbols is not None:
            least_symbols, least_freqs = parse_pairs(line)
            epochs[epoch] = (most_symbols, most_freqs, least_symbols, least_freqs)
            most_symbols = None
            most_freqs = None
    return epochs


def select_epoch(
    epoch_map: dict[int, tuple[list[str], list[float], list[str], list[float]]],
    target_epoch: int,
) -> tuple[list[str], list[float], list[str], list[float]] | None:
    return epoch_map.get(target_epoch)


def select_before_epoch(
    epoch_map: dict[int, tuple[list[str], list[float], list[str], list[float]]],
    cutoff_epoch: int,
) -> tuple[list[str], list[float], list[str], list[float]] | None:
    candidates = [e for e in epoch_map if e < cutoff_epoch]
    if not candidates:
        return None
    return epoch_map[max(candidates)]


def run_extract(script_path: Path, log_path: Path) -> str | None:
    proc = subprocess.run(
        [sys.executable, str(script_path), str(log_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            f"extract_epoch_glyphs.py failed for {log_path}: {proc.stderr.strip()}\n"
        )
        return None
    return proc.stdout


def collect_logs(root: Path) -> list[Path]:
    patterns = ["out_*.log"]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(root.glob(pat))
    if not paths:
        paths = list(root.glob("*.log"))
    return sorted(set(paths))


def parse_run_id(path: Path, fallback: int) -> tuple[str, int]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        run_id = match.group(1)
        return run_id, int(run_id)
    return path.stem, fallback


def build_ranks(top_symbols: list[str], bottom_symbols: list[str]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for idx, sym in enumerate(top_symbols[:TOP_N]):
        ranks.setdefault(sym, idx + 1)
    for idx, sym in enumerate(bottom_symbols[:BOTTOM_N]):
        ranks.setdefault(sym, -(idx + 1))
    return ranks


def overlap_count(a: set[str], b: set[str]) -> int:
    return len(a & b)


def rank_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    total = 0
    for sym in set(a) | set(b):
        total += a.get(sym, 0) * b.get(sym, 0)
    return total


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def collect_jaccards(runs_a: list[RunData], runs_b: list[RunData], attr: str) -> list[float]:
    values = []
    if runs_a is runs_b:
        for i in range(len(runs_a)):
            for j in range(i + 1, len(runs_a)):
                values.append(
                    overlap_count(
                        set(getattr(runs_a[i], attr)),
                        set(getattr(runs_a[j], attr)),
                    )
                )
    else:
        for a in runs_a:
            for b in runs_b:
                values.append(
                    overlap_count(set(getattr(a, attr)), set(getattr(b, attr)))
                )
    return values


def collect_rank_scores(runs_a: list[RunData], runs_b: list[RunData]) -> list[float]:
    values = []
    for a in runs_a:
        for b in runs_b:
            values.append(rank_similarity(a.ranks, b.ranks))
    return values


def write_table(rows: list[RunData], out_path: Path) -> None:
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "mode", "takeover_status", "collapse_index"])
        for row in rows:
            writer.writerow([row.run_id, row.mode, row.takeover_status, row.collapse_index])


def plot_results(
    output_dir: Path,
    jaccard_data: dict[str, dict[str, list[int]]],
    rank_data: dict[str, list[float]],
    runs: list[RunData],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("matplotlib not available, skipping plots.\n")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Overlap boxplots
    comp_labels = list(jaccard_data.keys())
    top_data = [jaccard_data[label]["top"] for label in comp_labels]
    bottom_data = [jaccard_data[label]["bottom"] for label in comp_labels]
    top_order = sorted(
        range(len(comp_labels)),
        key=lambda idx: statistics.mean(top_data[idx]) if top_data[idx] else 0.0,
        reverse=True,
    )
    bottom_order = sorted(
        range(len(comp_labels)),
        key=lambda idx: statistics.mean(bottom_data[idx]) if bottom_data[idx] else 0.0,
        reverse=True,
    )
    top_labels = [comp_labels[idx] for idx in top_order]
    bottom_labels = [comp_labels[idx] for idx in bottom_order]
    top_data = [top_data[idx] for idx in top_order]
    bottom_data = [bottom_data[idx] for idx in bottom_order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    font_size = 12
    title_size = 13
    tick_size = 10
    top_positions = [1 + i * 0.85 for i in range(len(top_data))]
    bottom_positions = [1 + i * 0.85 for i in range(len(bottom_data))]
    axes[0].boxplot(
        top_data, positions=top_positions, widths=0.65, showmeans=True
    )
    axes[0].set_xticks(top_positions)
    axes[0].set_xticklabels(top_labels, fontsize=tick_size)
    axes[0].set_title("Overlap count (Top-16)", fontsize=title_size)
    axes[0].set_ylabel("Overlap count", fontsize=font_size)
    axes[1].boxplot(
        bottom_data, positions=bottom_positions, widths=0.65, showmeans=True
    )
    axes[1].set_xticks(bottom_positions)
    axes[1].set_xticklabels(bottom_labels, fontsize=tick_size)
    axes[1].set_title("Overlap count (Bottom-16)", fontsize=title_size)
    axes[1].set_ylabel("")
    axes[0].tick_params(axis="x", labelrotation=20, labelsize=tick_size, pad=1)
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=tick_size, pad=1)
    axes[0].tick_params(axis="y", labelsize=tick_size)
    axes[1].tick_params(axis="y", labelsize=tick_size)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.88, wspace=0.12)
    fig.savefig(output_dir / "overlap_boxplots.png", dpi=150)
    plt.close(fig)

    # Rank stability bar plot
    rank_labels = list(rank_data.keys())
    rank_means = [statistics.mean(rank_data[label]) if rank_data[label] else 0.0 for label in rank_labels]
    rank_stds = [statistics.stdev(rank_data[label]) if len(rank_data[label]) > 1 else 0.0 for label in rank_labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rank_labels, rank_means, yerr=rank_stds, capsize=4)
    ax.set_title("Rank stability score (unnormalized)")
    ax.set_ylabel("Similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "rank_stability.png", dpi=150)
    plt.close(fig)

    # Collapse index scatter
    group_colors = {
        "R_null": "#1f77b4",
        "R_pre": "#ff7f0e",
        "R_post": "#d62728",
        "N": "#2ca02c",
    }
    status_map = {
        "R_null": "R_non_takeover",
        "R_pre": "R_before_takeover",
        "R_post": "R_takeover",
    }
    fig, ax = plt.subplots(figsize=(6, 6))
    max_index = max((r.run_index for r in runs), default=0)
    threshold = 20
    left_weight = 0.8
    right_weight = 0.2

    def scale_x(value: int) -> float:
        if max_index <= threshold:
            return float(value)
        if value < threshold:
            return left_weight * (value / (threshold - 1))
        right_span = max_index - threshold + 1
        return left_weight + right_weight * ((value - threshold + 1) / right_span)
    for group, color in group_colors.items():
        if group == "N":
            group_runs = [r for r in runs if r.mode == "N"]
        else:
            status = status_map[group]
            group_runs = [r for r in runs if r.takeover_status == status]
        xs = [scale_x(r.run_index) for r in group_runs]
        ys = [r.collapse_index for r in group_runs]
        if xs:
            ax.scatter(xs, ys, label=group, alpha=0.75, color=color)
    ax.set_title("Collapse index per run")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Collapse index (M / L)")
    if runs:
        first_range = list(range(0, min(20, max_index + 1)))
        ticks = first_range[:]
        if max_index >= 20:
            ticks.append(max_index)
        ax.set_xticks([scale_x(i) for i in ticks])
        ax.set_xticklabels([str(i) for i in ticks])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "collapse_index.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze epoch glyph statistics.")
    parser.add_argument(
        "--interaction-dir",
        default="runs/interaction",
        help="Path to interaction logs",
    )
    parser.add_argument(
        "--random-dir",
        default="runs/random",
        help="Path to random logs",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=TARGET_EPOCH,
        help="Epoch to analyze",
    )
    parser.add_argument(
        "--plots-dir",
        default="analysis_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--table-path",
        default="analysis_epoch_glyphs.csv",
        help="CSV output path for per-run table",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve().parent / "extract_epoch_glyphs.py"
    interaction_root = Path(args.interaction_dir)
    random_root = Path(args.random_dir)

    runs: list[RunData] = []
    fallback_index = 0

    for mode, root in [("R", interaction_root), ("N", random_root)]:
        for log_path in collect_logs(root):
            output = run_extract(script_path, log_path)
            if output is None:
                continue
            epoch_map = parse_all_epochs(output)
            parsed = select_epoch(epoch_map, args.epoch)
            if parsed is None:
                sys.stderr.write(f"Epoch {args.epoch} not found in {log_path}\n")
                continue
            top_syms, top_freqs, bottom_syms, bottom_freqs = parsed
            top_syms = top_syms[:TOP_N]
            top_freqs = top_freqs[:TOP_N]
            bottom_syms = bottom_syms[:BOTTOM_N]
            bottom_freqs = bottom_freqs[:BOTTOM_N]
            run_id, run_index = parse_run_id(log_path, fallback_index)
            fallback_index += 1

            if mode == "R":
                try:
                    run_id_int = int(run_id)
                except ValueError:
                    run_id_int = None
                takeover_status = (
                    "R_non_takeover"
                    if run_id_int in R_NON_TAKEOVER
                    else "R_takeover"
                )
            else:
                takeover_status = "N"

            mean_top = statistics.mean(top_freqs) if top_freqs else 0.0
            mean_bottom = statistics.mean(bottom_freqs) if bottom_freqs else 0.0
            collapse_index = math.inf if mean_bottom == 0 else mean_top / mean_bottom

            ranks = build_ranks(top_syms, bottom_syms)

            runs.append(
                RunData(
                    run_id=run_id,
                    mode=mode,
                    takeover_status=takeover_status,
                    top_symbols=top_syms,
                    bottom_symbols=bottom_syms,
                    top_freqs=top_freqs,
                    bottom_freqs=bottom_freqs,
                    ranks=ranks,
                    collapse_index=collapse_index,
                    run_index=run_index,
                )
            )

            if mode == "R" and run_id_int in R_TAKEOVER_EPOCHS:
                before = select_before_epoch(
                    epoch_map, R_TAKEOVER_EPOCHS[run_id_int]
                )
                if before is None:
                    sys.stderr.write(
                        f"No epoch before takeover for run {run_id} in {log_path}\n"
                    )
                    continue
                top_syms, top_freqs, bottom_syms, bottom_freqs = before
                top_syms = top_syms[:TOP_N]
                top_freqs = top_freqs[:TOP_N]
                bottom_syms = bottom_syms[:BOTTOM_N]
                bottom_freqs = bottom_freqs[:BOTTOM_N]
                mean_top = statistics.mean(top_freqs) if top_freqs else 0.0
                mean_bottom = statistics.mean(bottom_freqs) if bottom_freqs else 0.0
                collapse_index = math.inf if mean_bottom == 0 else mean_top / mean_bottom
                ranks = build_ranks(top_syms, bottom_syms)
                runs.append(
                    RunData(
                        run_id=f"{run_id}_before",
                        mode=mode,
                        takeover_status="R_before_takeover",
                        top_symbols=top_syms,
                        bottom_symbols=bottom_syms,
                        top_freqs=top_freqs,
                        bottom_freqs=bottom_freqs,
                        ranks=ranks,
                        collapse_index=collapse_index,
                        run_index=run_index,
                    )
                )

    if not runs:
        sys.stderr.write("No runs parsed successfully.\n")
        return 1

    group_map = {
        "R_null": [r for r in runs if r.takeover_status == "R_non_takeover"],
        "R_post": [r for r in runs if r.takeover_status == "R_takeover"],
        "R_pre": [r for r in runs if r.takeover_status == "R_before_takeover"],
        "N": [r for r in runs if r.mode == "N"],
    }

    jaccard_specs = {
        "R_null vs R_null": ("R_null", "R_null"),
        "R_post vs R_post": ("R_post", "R_post"),
        "R_pre vs R_pre": ("R_pre", "R_pre"),
        "R_null vs N": ("R_null", "N"),
        "R_post vs N": ("R_post", "N"),
        "R_pre vs N": ("R_pre", "N"),
        "R_pre vs R_null": ("R_pre", "R_null"),
        "N vs N": ("N", "N"),
    }

    jaccard_data: dict[str, dict[str, list[int]]] = {}
    for label, (left, right) in jaccard_specs.items():
        runs_a = group_map[left]
        runs_b = group_map[right]
        jaccard_data[label] = {
            "top": collect_jaccards(runs_a, runs_b, "top_symbols"),
            "bottom": collect_jaccards(runs_a, runs_b, "bottom_symbols"),
        }

    rank_specs = {
        "R_null vs N": ("R_null", "N"),
        "R_post vs N": ("R_post", "N"),
    }
    rank_data: dict[str, list[float]] = {}
    for label, (left, right) in rank_specs.items():
        rank_data[label] = collect_rank_scores(group_map[left], group_map[right])

    table_path = Path(args.table_path)
    write_table(runs, table_path)

    print("Per-run table written to:", table_path)
    print("")
    print("Overlap counts (mean ± std, normalized by 16 in parentheses):")
    labels = list(jaccard_data.keys())
    top_order = sorted(
        labels,
        key=lambda l: statistics.mean(jaccard_data[l]["top"])
        if jaccard_data[l]["top"]
        else 0.0,
        reverse=True,
    )
    bottom_order = sorted(
        labels,
        key=lambda l: statistics.mean(jaccard_data[l]["bottom"])
        if jaccard_data[l]["bottom"]
        else 0.0,
        reverse=True,
    )
    print("Top (sorted high → low):")
    for label in top_order:
        vals = jaccard_data[label]
        top_mean, top_std = summarize(vals["top"])
        top_norm_mean = top_mean / TOP_N
        top_norm_std = top_std / TOP_N
        print(
            f"- {label} | top: {top_mean:.2f} ± {top_std:.2f} "
            f"(norm {top_norm_mean:.4f} ± {top_norm_std:.4f}, n={len(vals['top'])})"
        )
    print("Bottom (sorted high → low):")
    for label in bottom_order:
        vals = jaccard_data[label]
        bottom_mean, bottom_std = summarize(vals["bottom"])
        bottom_norm_mean = bottom_mean / BOTTOM_N
        bottom_norm_std = bottom_std / BOTTOM_N
        print(
            f"- {label} | bottom: {bottom_mean:.2f} ± {bottom_std:.2f} "
            f"(norm {bottom_norm_mean:.4f} ± {bottom_norm_std:.4f}, n={len(vals['bottom'])})"
        )

    print("")
    print("Rank stability (unnormalized + normalized by max):")
    for label, vals in rank_data.items():
        mean_val, std_val = summarize(vals)
        norm_mean = mean_val / MAX_RANK_SCORE if MAX_RANK_SCORE else 0.0
        norm_std = std_val / MAX_RANK_SCORE if MAX_RANK_SCORE else 0.0
        print(
            f"- {label}: {mean_val:.2f} ± {std_val:.2f} "
            f"(normalized {norm_mean:.4f} ± {norm_std:.4f}, n={len(vals)})"
        )

    print("")
    print("Collapse index (mean ± std):")
    all_collapse = [r.collapse_index for r in runs]
    mean_all, std_all = summarize(all_collapse)
    print(f"- All runs: {mean_all:.4f} ± {std_all:.4f} (n={len(all_collapse)})")
    for label, group in group_map.items():
        values = [r.collapse_index for r in group]
        mean_val, std_val = summarize(values)
        print(f"- {label}: {mean_val:.4f} ± {std_val:.4f} (n={len(values)})")

    plot_results(Path(args.plots_dir), jaccard_data, rank_data, runs)
    print("")
    print("Plots written to:", Path(args.plots_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
