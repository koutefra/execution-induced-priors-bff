#!/usr/bin/env python3
"""
Analyze output of report_replicators.py from stdin.

- Counts replicator events (score >= MIN_SCORE only)
- Classifies them into:
    (A) palindromic replicators
    (B) offset replicators
    (C) canonical 9-symbol replicators
    (D) 10-symbol replicators
    (E) other replicators
- Prints up to 10 examples from each class (opcode strips only)
"""

import random
import re
import sys

COPY_OPS = {".", ","}
H0_OPS = {"<", ">"}
H1_OPS = {"{", "}"}
INC_DEC_OPS = {"+", "-"}
OP_SYMBOLS = {"[", "]", "+", "-", ".", ",", "<", ">", "{", "}", "0"}

MAX_EXAMPLES = 10
MIN_SCORE = 60
SCORE_RE = re.compile(r"score=(\d+)")


def opposite_dirs(h0: str, h1: str) -> bool:
    return (h0 == "<" and h1 == "}") or (h0 == ">" and h1 == "{")


def cleanse(ops: list[str]) -> list[str]:
    cancel_pairs = {(">", "<"), ("}", "{"), ("<", ">"), ("{", "}")}
    res: list[str] = []

    for op in ops:
        if op == '0':
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
    if len(body) not in {5, 6}:
        return False

    loop_start = ["[" for c in body if c == "["]
    copies = [c for c in body if c in COPY_OPS]
    h0s = [c for c in body if c in H0_OPS]
    h1s = [c for c in body if c in H1_OPS]
    loop_end = ["]" for c in body if c == "]"]
    inc_decs = [c for c in body if c in INC_DEC_OPS]

    if not (
        len(loop_start) == len(copies) == len(h0s) == len(h1s) == len(loop_end) == 1
    ):
        return False

    if len(body) == 5:
        return len(inc_decs) == 0 and opposite_dirs(h0s[0], h1s[0])

    if len(inc_decs) != 1 or copies[0] != ",":
        return False
    if not any(body[i] == "," and body[i - 1] in INC_DEC_OPS for i in range(1, len(body))):
        return False

    return opposite_dirs(h0s[0], h1s[0])

    
def find_reverse_replicator(ops: list[str]) -> tuple[int, int]:
    # returns the indices of the found replicator, if exists, otherwise None
    n = len(ops)
    for i in range(n):
        if ops[i] != "[":
            continue

        body_start = i
        for body_len in (5, 6):
            body_end = body_start + body_len
            if body_end >= n:
                continue

            if not is_valid_body(ops[body_start:body_end]):
                continue

            return body_start, body_end

    return None


def is_palindromic(raw_tape: str) -> bool:
    tape = list(raw_tape)
    if len(tape) % 2 != 0:
        return False

    half = len(tape) // 2
    left = tape[:half]
    right = tape[half:]
    if right != list(reversed(left)):
        return False

    left_ops = cleanse([c for c in left if c in OP_SYMBOLS])
    return find_reverse_replicator(left_ops) is not None


def strip_zeros(ops: list[str]) -> tuple[list[str], list[int]]:
    non_zero_ops = []
    index_map = []
    for idx, op in enumerate(ops):
        if op == "0":
            continue
        non_zero_ops.append(op)
        index_map.append(idx)
    return non_zero_ops, index_map


def is_offset_replicator(raw_ops: list[str]) -> bool:
    non_zero_ops, index_map = strip_zeros(raw_ops)
    n = len(non_zero_ops)

    for i in range(n):
        if non_zero_ops[i] != "[":
            continue

        j = i + 1
        if j >= n or non_zero_ops[j] not in H0_OPS:
            continue

        direction = non_zero_ops[j]
        while j < n and non_zero_ops[j] == direction:
            j += 1
        if j >= n or non_zero_ops[j] != "]":
            continue

        body_start = j + 1
        if body_start >= n:
            continue

        for body_len in (5, 6):
            body_end = body_start + body_len
            if body_end > n:
                continue

            if not is_valid_body(non_zero_ops[body_start:body_end]):
                continue

            last_body_op_idx = index_map[body_end - 1]
            if "0" not in raw_ops[last_body_op_idx + 1:]:
                continue

            return True

    return False


def clasify(ops: list[str]) -> str:
    n = len(ops)
    body_1 = find_reverse_replicator(ops)
    body_2_reversed = find_reverse_replicator(list(reversed(ops)))
    body_2 = [n - i  for i in body_2_reversed] if body_2_reversed else None
    body_2 = tuple(reversed(body_2)) if body_2_reversed else None
    if not body_1 or not body_2:
        return None
    elif body_1[1] == body_2[0]:
        return "nine"

    return "ten"


def classify_replicator(ops: list[str], raw_ops: list[str], raw_tape: str) -> str | None:
    rep_type = clasify(ops)
    if rep_type == "ten" and is_palindromic(raw_tape):
        return "palindromic"
    if is_offset_replicator(raw_ops):
        return "offset"
    return rep_type


def main() -> None:
    lines = [l.strip() for l in sys.stdin.read().splitlines() if l.strip()]

    total = 0
    palindromic = 0
    offset = 0
    nine_symbol = 0
    ten_symbol = 0

    examples_palindromic = []
    examples_offset = []
    examples_9 = []
    examples_10 = []
    examples_other = []
    seen_palindromic = 0
    seen_offset = 0
    seen_9 = 0
    seen_10 = 0
    seen_other = 0

    for block_start in range(0, len(lines), 4):
        if block_start + 3 >= len(lines):
            break

        header = lines[block_start + 1]
        full_tape = lines[block_start + 2]
        ops_line = lines[block_start + 3]
        match = SCORE_RE.search(header)
        if not match:
            continue
        score = int(match.group(1))
        if score < MIN_SCORE:
            continue

        raw_ops = ops_line.split()
        ops = cleanse(raw_ops)
        total += 1

        rep_type = classify_replicator(ops, raw_ops, full_tape)

        is_palindromic = rep_type == "palindromic"
        is_offset = rep_type == "offset"
        is_nine = rep_type == "nine"
        is_ten = rep_type == "ten"

        if is_palindromic:
            palindromic += 1
            seen_palindromic += 1
            if len(examples_palindromic) < MAX_EXAMPLES:
                examples_palindromic.append((full_tape, ops_line))
            else:
                pick = random.randrange(seen_palindromic)
                if pick < MAX_EXAMPLES:
                    examples_palindromic[pick] = (full_tape, ops_line)
        if is_offset:
            offset += 1
            seen_offset += 1
            if len(examples_offset) < MAX_EXAMPLES:
                examples_offset.append((full_tape, ops_line))
            else:
                pick = random.randrange(seen_offset)
                if pick < MAX_EXAMPLES:
                    examples_offset[pick] = (full_tape, ops_line)
        if is_nine:
            nine_symbol += 1
            seen_9 += 1
            if len(examples_9) < MAX_EXAMPLES:
                examples_9.append((full_tape, ops_line))
            else:
                pick = random.randrange(seen_9)
                if pick < MAX_EXAMPLES:
                    examples_9[pick] = (full_tape, ops_line)
        if is_ten:
            ten_symbol += 1
            seen_10 += 1
            if len(examples_10) < MAX_EXAMPLES:
                examples_10.append((full_tape, ops_line))
            else:
                pick = random.randrange(seen_10)
                if pick < MAX_EXAMPLES:
                    examples_10[pick] = (full_tape, ops_line)
        if not (is_palindromic or is_offset or is_nine or is_ten):
            seen_other += 1
            if len(examples_other) < MAX_EXAMPLES:
                examples_other.append((full_tape, ops_line))
            else:
                pick = random.randrange(seen_other)
                if pick < MAX_EXAMPLES:
                    examples_other[pick] = (full_tape, ops_line)

    print("Total replicator events:", total)
    print("Palindromic replicators:", palindromic)
    print("Offset replicators:", offset)
    print("9-symbol replicators:", nine_symbol)
    print("10-symbol replicators:", ten_symbol)
    print(
        "Other replicators:",
        total - palindromic - offset - nine_symbol - ten_symbol,
    )

    print("\n=== Examples: palindromic replicators ===")
    for full_tape, ops_line in examples_palindromic:
        print(full_tape)
        print(ops_line)

    print("\n=== Examples: offset replicators ===")
    for full_tape, ops_line in examples_offset:
        print(full_tape)
        print(ops_line)

    print("\n=== Examples: 9-symbol replicators ===")
    for full_tape, ops_line in examples_9:
        print(full_tape)
        print(ops_line)

    print("\n=== Examples: 10-symbol replicators ===")
    for full_tape, ops_line in examples_10:
        print(full_tape)
        print(ops_line)

    print("\n=== Examples: other replicators ===")
    for full_tape, ops_line in examples_other:
        print(full_tape)
        print(ops_line)



if __name__ == "__main__":
    main()
