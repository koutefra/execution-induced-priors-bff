#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;:]*[@-~]")
HEADER_RE = re.compile(r"Epochs:\s+(\d+)")
PAIR_RE = re.compile(r"(\S+)\s+([0-9]*\.?[0-9]+)%")
CHAR_TABLE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_pairs(line: str):
    return [(m.group(1), m.group(2)) for m in PAIR_RE.finditer(line)]


def decode_c_string(value: str) -> str:
    return bytes(value, "utf-8").decode("unicode_escape")


def load_bff_glyph_map(bff_inc_path: Path):
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
    glyphs = glyphs[:256]

    glyph_to_byte = {"0": 0}
    for ch in "[]+-.,<>{}":
        glyph_to_byte[ch] = ord(ch)
    for idx, glyph in enumerate(glyphs):
        glyph_to_byte.setdefault(glyph, idx)
    return glyph_to_byte


def format_pairs(pairs, glyph_map):
    parts = []
    for glyph, pct in pairs:
        byte_val = glyph_map.get(glyph)
        if byte_val is None:
            parts.append(f"{glyph}(?)={pct}%")
        else:
            parts.append(f"{glyph}({byte_val})={pct}%")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract most/least frequent glyphs per epoch from out_*.log files."
        )
    )
    parser.add_argument("log_path", help="Path to out_*.log file")
    default_bff_inc = Path(__file__).resolve().parent.parent / "bff.inc.h"
    parser.add_argument(
        "--bff-inc",
        default=str(default_bff_inc),
        help="Path to bff.inc.h for glyph-to-byte mapping",
    )
    args = parser.parse_args()

    epoch = None
    most = None
    state = "seek_header"
    try:
        glyph_map = load_bff_glyph_map(Path(args.bff_inc))
    except (OSError, ValueError) as exc:
        print(f"Failed to load glyph map: {exc}", file=sys.stderr)
        return 1

    try:
        with open(args.log_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = strip_ansi(raw_line).strip()
                if not line:
                    continue

                if "Elapsed:" in line:
                    match = HEADER_RE.search(line)
                    if match:
                        epoch = int(match.group(1))
                        state = "seek_most"
                        most = None
                    continue

                if state == "seek_most":
                    if "%" in line:
                        pairs = parse_pairs(line)
                        if pairs:
                            most = pairs
                            state = "seek_least"
                    continue

                if state == "seek_least":
                    if "%" in line:
                        pairs = parse_pairs(line)
                        if pairs:
                            if epoch is not None and most is not None:
                                print(f"Epoch {epoch}")
                                print(f"most: {format_pairs(most, glyph_map)}")
                                print(f"least: {format_pairs(pairs, glyph_map)}")
                                print("")
                            state = "seek_header"
                    continue
    except FileNotFoundError:
        print(f"File not found: {args.log_path}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
