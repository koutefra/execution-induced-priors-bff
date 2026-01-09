#!/usr/bin/env python3
"""
Scan runs/<mode> logs and report, per log file, the first replicator
with score >= threshold (decoded).
"""

import argparse
import ast
import csv
import math
import pathlib
import statistics


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

OP_SYMBOLS = set(COMMAND_BYTES.values()) | {"0"}
COPY_OPS = {".", ","}
H0_OPS = {"<", ">"}
H1_OPS = {"{", "}"}


def map_byte(byte: int) -> str:
    if byte == 0:
        return "0"
    if byte in COMMAND_BYTES:
        return COMMAND_BYTES[byte]
    return chr(0x0100 + byte)


def decode_hex_tape(hex_string: str) -> list[str]:
    tape_bytes = bytes.fromhex(hex_string.strip())
    return [map_byte(b) for b in tape_bytes]


def opposite_dirs(h0: str, h1: str) -> bool:
    return (h0 == "<" and h1 == "}") or (h0 == ">" and h1 == "{")


def cleanse(ops: list[str]) -> list[str]:
    cancel_pairs = {(">", "<"), ("}", "{"), ("<", ">"), ("{", "}")}
    res: list[str] = []

    for op in ops:
        if op == "0":
            continue

        if res:
            last = res[-1]
            if op in COPY_OPS and last in COPY_OPS:
                continue
            if (last, op) in cancel_pairs:
                res.pop()
                continue
        res.append(op)

    return res


def is_valid_body(body: list[str]) -> bool:
    if len(body) != 5:
        return False

    loop_start = ["[" for c in body if c == "["]
    copies = [c for c in body if c in COPY_OPS]
    h0s = [c for c in body if c in H0_OPS]
    h1s = [c for c in body if c in H1_OPS]
    loop_end = ["]" for c in body if c == "]"]

    if not (
        len(loop_start) == len(copies) == len(h0s) == len(h1s) == len(loop_end) == 1
    ):
        return False

    return opposite_dirs(h0s[0], h1s[0])


def find_reverse_replicator(ops: list[str]) -> tuple[int, int] | None:
    n = len(ops)
    for i in range(n):
        if ops[i] != "[":
            continue

        body_start = i
        body_end = body_start + 5
        if body_end >= n:
            continue

        if not is_valid_body(ops[body_start:body_end]):
            continue

        return body_start, body_end

    return None


def classify_replicator(ops: list[str]) -> str | None:
    n = len(ops)
    body_1 = find_reverse_replicator(ops)
    body_2_reversed = find_reverse_replicator(list(reversed(ops)))
    body_2 = [n - i for i in body_2_reversed] if body_2_reversed else None
    body_2 = tuple(reversed(body_2)) if body_2_reversed else None
    if not body_1 or not body_2:
        return None
    if body_1[1] == body_2[0]:
        return "nine"

    return "ten"


def find_first_per_log(root: pathlib.Path, threshold: int):
    results = []
    for log_path in sorted(root.glob("log_*.log")):
        found = None
        with log_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if int(row.get("number_selfreps", 0)) == 0:
                    continue
                scores = ast.literal_eval(row["selfrep_scores"])
                for idx, score in enumerate(scores):
                    if int(score) >= threshold:
                        tapes = ast.literal_eval(row["selfrep_tapes"])
                        hex_tape = tapes[idx]
                        decoded = decode_hex_tape(hex_tape)
                        full_tape = "".join(decoded)
                        ops = [c for c in decoded if c in OP_SYMBOLS]
                        rep_class = classify_replicator(cleanse(ops))
                        found = {
                            "log": log_path,
                            "epoch": int(row["epoch"]),
                            "rep_index": idx,
                            "score": int(score),
                            "tape": hex_tape,
                            "decoded": full_tape,
                            "ops": ops,
                            "class": rep_class or "other",
                        }
                        break
                if found:
                    break
        if found:
            results.append(found)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Report first replicator with score >= threshold."
    )
    parser.add_argument("mode", choices=["interaction", "random", "reinit"])
    parser.add_argument("--threshold", type=int, default=60)
    args = parser.parse_args()

    root = pathlib.Path("runs") / args.mode
    if not root.is_dir():
        raise SystemExit(f"Missing folder: {root}")

    results = find_first_per_log(root, args.threshold)
    if not results:
        print(f"No replicator with score >= {args.threshold} in {root}")
        return

    for result in results:
        print("=" * 80)
        print(f"log={result['log']}")
        print(f"epoch={result['epoch']}")
        print(f"rep_index={result['rep_index']}")
        print(f"score={result['score']}")
        print(f"class={result['class']}")
        print(result["decoded"])
        print("   ".join(result["ops"]))

    epochs = [result["epoch"] for result in results]
    classes = [result["class"] for result in results]
    class_counts = {
        "nine": sum(1 for c in classes if c == "nine"),
        "ten": sum(1 for c in classes if c == "ten"),
        "other": sum(1 for c in classes if c == "other"),
    }
    print("=" * 80)
    print("epochs=" + ", ".join(str(epoch) for epoch in epochs))
    print(
        "class_counts="
        + ", ".join(f"{name}:{count}" for name, count in class_counts.items())
    )
    mean_epoch = statistics.mean(epochs)
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    if len(epochs) > 1:
        stdev_epoch = statistics.stdev(epochs)
        stderr = stdev_epoch / math.sqrt(len(epochs))
        t_critical_975 = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            11: 2.201,
            12: 2.179,
            13: 2.160,
            14: 2.145,
            15: 2.131,
            16: 2.120,
            17: 2.110,
            18: 2.101,
            19: 2.093,
            20: 2.086,
            21: 2.080,
            22: 2.074,
            23: 2.069,
            24: 2.064,
            25: 2.060,
            26: 2.056,
            27: 2.052,
            28: 2.048,
            29: 2.045,
            30: 2.042,
        }
        df = len(epochs) - 1
        t_value = t_critical_975.get(df, 1.96)
        ci_half_width = t_value * stderr
        ci_low = mean_epoch - ci_half_width
        ci_high = mean_epoch + ci_half_width
        ci_text = f"[{ci_low:.3f}, {ci_high:.3f}]"
    else:
        ci_text = "n/a"
    print(f"mean_epoch={mean_epoch:.3f}")
    print(f"min_epoch={min_epoch}")
    print(f"max_epoch={max_epoch}")
    print(f"mean_95pct_ci={ci_text}")


if __name__ == "__main__":
    main()
