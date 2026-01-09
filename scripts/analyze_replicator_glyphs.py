#!/usr/bin/env python3
"""
Analyze replicator glyph frequencies at a target epoch range for specific runs.

For each log_*.log in runs/interaction with a run id in the configured set:
- load replicator tapes in an epoch range ending at 15872
- decode the 5 replicator tapes to glyphs
- compute average symbol counts and percentages
- print top 16 symbols per run
- print mean higher_entropy over a fixed epoch range
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

CHAR_TABLE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
COMMAND_BYTES = {
    ord("["): "[",
    ord("]"): "]",
    ord("+"): "+",
    ord("-"): "-",
    ord("."): ".",
    ord(","): ",",
    ord("<"): "<",
    ord(">"): ">",
    ord("{"): "{",
    ord("}"): "}",
}
DEFAULT_RUNS = [1, 2, 8, 10, 12, 18]


def decode_c_string(value: str) -> str:
    return bytes(value, "utf-8").decode("unicode_escape")


def load_bff_glyphs(bff_inc_path: Path) -> list[str]:
    text = bff_inc_path.read_text(encoding="utf-8", errors="replace")
    start = text.find("static const char *data[256] = {")
    if start == -1:
        raise ValueError(f"CharacterRepr table not found in {bff_inc_path}")
    block = text[start:]
    block = block[block.find("{") + 1 : block.find("};")]
    raw = CHAR_TABLE_RE.findall(block)
    glyphs = [decode_c_string(s) for s in raw]
    if len(glyphs) < 256:
        raise ValueError(
            f"CharacterRepr table too small ({len(glyphs)} entries) in {bff_inc_path}"
        )
    return glyphs[:256]


def parse_hex_list(value: str) -> list[str] | None:
    try:
        values = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(v, str) for v in values):
        return None
    normalized = ["".join(v.split()) for v in values]
    if not all(v and len(v) % 2 == 0 for v in normalized):
        return None
    return normalized


def load_epoch_data_range(
    log_path: Path,
    tapes_start: int,
    tapes_end: int,
    entropy_start: int,
    entropy_end: int,
) -> tuple[list[str], list[float]] | None:
    tapes: list[str] = []
    entropy_values: list[float] = []
    try:
        with log_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    epoch = int(row.get("epoch", ""))
                except ValueError:
                    continue
                if tapes_start <= epoch <= tapes_end:
                    parsed = parse_hex_list(row.get("selfrep_tapes", ""))
                    if parsed:
                        tapes.extend(parsed)
                if entropy_start <= epoch <= entropy_end:
                    try:
                        entropy_values.append(float(row.get("higher_entropy", "")))
                    except ValueError:
                        continue
    except FileNotFoundError:
        return None
    return tapes, entropy_values


def map_byte(byte: int, glyphs: list[str]) -> str:
    if byte == 0:
        return "0"
    if byte in COMMAND_BYTES:
        return COMMAND_BYTES[byte]
    return glyphs[byte]


def decode_hex_tape(hex_string: str, glyphs: list[str]) -> list[str]:
    data = bytes.fromhex(hex_string.strip())
    return [map_byte(b, glyphs) for b in data]


def compute_stats(
    hex_tapes: list[str], glyphs: list[str], all_symbols: list[str]
) -> list[tuple[str, float, float]]:
    total_counts: dict[str, int] = {sym: 0 for sym in all_symbols}
    total_lengths = 0
    replicator_count = 0

    for hex_tape in hex_tapes:
        symbols = decode_hex_tape(hex_tape, glyphs)
        for sym in symbols:
            total_counts[sym] += 1
        total_lengths += len(symbols)
        replicator_count += 1

    if replicator_count == 0:
        return []

    avg_len = total_lengths / replicator_count
    stats = []
    for sym, cnt in total_counts.items():
        avg_count = cnt / replicator_count
        pct = 0.0 if avg_len == 0 else (avg_count / avg_len) * 100.0
        stats.append((sym, avg_count, pct))
    stats.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return stats


def parse_run_ids(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    run_ids = []
    for part in parts:
        run_ids.append(int(part))
    return run_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report top glyphs for replicators at a target epoch."
    )
    parser.add_argument(
        "--interaction-dir",
        default="runs/interaction",
        help="Directory containing log_*.log files",
    )
    parser.add_argument(
        "--epoch-end",
        type=int,
        default=15872,
        help="End epoch to extract (inclusive)",
    )
    parser.add_argument(
        "--epoch-span",
        type=int,
        default=512,
        help="Number of epochs before end to include (inclusive)",
    )
    parser.add_argument(
        "--entropy-start",
        type=int,
        default=15360,
        help="Start epoch for higher_entropy mean (inclusive)",
    )
    parser.add_argument(
        "--entropy-end",
        type=int,
        default=15672,
        help="End epoch for higher_entropy mean (inclusive)",
    )
    parser.add_argument("--top", type=int, default=16, help="Top-N symbols to report")
    parser.add_argument(
        "--runs",
        default=",".join(str(x) for x in DEFAULT_RUNS),
        help="Comma-separated run ids to process",
    )
    default_bff_inc = Path(__file__).resolve().parent.parent / "bff.inc.h"
    parser.add_argument(
        "--bff-inc",
        default=str(default_bff_inc),
        help="Path to bff.inc.h for glyph mapping",
    )
    args = parser.parse_args()

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    try:
        glyphs = load_bff_glyphs(Path(args.bff_inc))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Failed to load glyph map: {exc}\n")
        return 1

    glyph_to_byte: dict[str, int] = {"0": 0}
    for ch in "[]+-.,<>{}":
        glyph_to_byte[ch] = ord(ch)
    for idx, glyph in enumerate(glyphs):
        glyph_to_byte.setdefault(glyph, idx)
    all_symbols = list(glyph_to_byte.keys())

    run_ids = parse_run_ids(args.runs)
    root = Path(args.interaction_dir)
    for run_id in run_ids:
        log_path = root / f"log_{run_id}.log"
        start_epoch = args.epoch_end - args.epoch_span
        data = load_epoch_data_range(
            log_path,
            start_epoch,
            args.epoch_end,
            args.entropy_start,
            args.entropy_end,
        )
        if data is None:
            sys.stderr.write(f"Missing log file: {log_path}\n")
            continue
        tapes, entropy_values = data
        if not tapes:
            sys.stderr.write(
                f"No replicators between epochs {start_epoch} and {args.epoch_end} in {log_path}\n"
            )
            continue
        stats = compute_stats(tapes, glyphs, all_symbols)
        if entropy_values:
            mean_entropy = sum(entropy_values) / len(entropy_values)
        else:
            mean_entropy = float("nan")
        print(
            f"Run {run_id} top {args.top} (epochs {start_epoch}..{args.epoch_end})"
        )
        for sym, _avg_count, pct in stats[: args.top]:
            byte = glyph_to_byte.get(sym)
            label = f"{sym}({byte})" if byte is not None else sym
            print(f"{label}={pct:.2f}%")
        print(f"Run {run_id} bottom {args.top} (epochs {start_epoch}..{args.epoch_end})")
        for sym, _avg_count, pct in stats[-args.top :]:
            byte = glyph_to_byte.get(sym)
            label = f"{sym}({byte})" if byte is not None else sym
            print(f"{label}={pct:.2f}%")
        if math.isnan(mean_entropy):
            print(
                f"Mean higher_entropy ({args.entropy_start}..{args.entropy_end}): n/a"
            )
        else:
            print(
                f"Mean higher_entropy ({args.entropy_start}..{args.entropy_end}): {mean_entropy:.6f}"
            )
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
