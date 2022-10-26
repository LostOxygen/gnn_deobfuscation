#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import string

"""
This script parses data from a diverse set of tools and unifies the format of MBAs to (adequately-named) files.
The following assumptions are guaranteed to hold:
* each line (an entry) follows exactly the following schema: `complex expression, simplified version`
* Complex/simplified expressions may contain whitespace but necessarily so
* each file contains at least one entry
* each file contains only entries using the *same* variables (e.g., x and y)
"""

# Directory where we will store out processed MBA files
OUT_DIR = Path("datasets")

# Source directories from which to pull MBAs
LOKI_PATH = Path("/tmp/temp/loki/experiments/experiment_10_mba_formula/data")
NEUREDUCE_PATH = Path("/tmp/temp/NeuReduce/dataset/linear")
MBABLAST_PATH = Path("/tmp/temp/MBA-Blast/dataset/")
MBAOBFUSCATOR_PATH = Path("/tmp/temp/MBA-Obfuscator/mba_obfuscator/dataset/")

# Output directory for unified data
UNIFIED_OUT_DIR = OUT_DIR / "unified"

# set of variable characters we can handle
VARS_SET = set(string.ascii_lowercase)
KNOWN_VARS = {"x", "y", "z", "t", "s"}
KNOWN_OPS = {"+", "-", "*", "&", "|", "^"}


def save(file_: Path, data: Iterable[str]) -> None:
    with open(file_, "w", encoding="utf-8") as f:
        f.writelines(sorted(list(data), key=lambda e: len(e)))


def get_vars(line: str) -> Set[str]:
    return VARS_SET.intersection(set(line))


def clean_line(line: str) -> Optional[str]:
    """
    Clean a line by removing artifacts such as trailing ,True.
    This will return None if a line resembles a comment (starts with '#')
    """
    if line.strip().startswith("#"):
        return None
    line = line.strip().rstrip("True").rstrip("False").strip().rstrip(",") + "\n"
    return line


def sort_variables(l: Iterable[str]) -> List[str]:
    """
    sort variables to fit the "weird", non-alphabetic sorting according to
    ["x", "y", "z", "t", "s"]
    """
    l = sorted(l)
    for c in ["t", "s"]:
        if c in l and len(l) > 1:
            l.remove(c)
            l += [c]
    return l


def unify_vars(line: str) -> Tuple[str, Set[str]]:
    """
    Convert a line such that a unified set of variables is used, namely:
    1 var : x
    2 vars: x, y
    3 vars: x, y, z
    4 vars: x, y, z, t
    5 vars: x, y, z, t, s

    5 or more vars found are not supported (ValueError)
    """
    vars = get_vars(line)
    # check if we found more variables than we can handle
    if len(vars) > len(KNOWN_VARS):
        raise ValueError(f"Can handle only up to {len(KNOWN_VARS)} but found: {len(vars)}: {vars}")
    # adapt our expectated list of variables to their number
    expected_vars = sort_variables(KNOWN_VARS)[:len(vars)]

    svars = sort_variables(vars.difference(expected_vars))
    replacement_vars = sort_variables(set(expected_vars).difference(vars))
    # print(svars, replacement_vars)
    assert len(svars) == len(replacement_vars)
    for i, var in enumerate(svars):
        if var != replacement_vars[i]:
            # print(f"replacing {var} by {replacement_vars[i]}")
            line = line.replace(var, replacement_vars[i])
    new_vars = get_vars(line)
    assert new_vars == set(expected_vars), \
                        f"Set mismatch: found {vars} but expected {expected_vars}"
    return line, new_vars


def neureduce() -> None:
    """
    NeuReduce: Reducing Mixed Boolean-Arithmetic Expressions by Recurrent Neural Network
    https://github.com/fvrmatteo/NeuReduce/ (original repository was removed)
    """
    neureduce_out_dir = OUT_DIR / "neudreduce"
    if neureduce_out_dir.exists():
        print(f"NeuReduce out dir exists at {neureduce_out_dir.as_posix()} -- skipping NeuReduce")
        return

    vars_to_data: Dict[int, Set[str]] = defaultdict(set)
    for file_ in NEUREDUCE_PATH.glob("**/*.csv"):
        if file_.name.startswith("."):
            continue
        print(f"Processing {file_}")
        with open(file_, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if (line := clean_line(line)) is not None:
                    line, vars = unify_vars(line)
                    assert line.count(",") == 1, f"NeuReduce {file_.as_posix()}: Found {line.count(',')} commas in line: {line}"
                    vars_to_data[len(vars)].add(line)
    # save data
    neureduce_out_dir.mkdir()
    for num_vars, data in vars_to_data.items():
        print(f"NeuReduce: Saving {len(data)} for {num_vars} variables")
        save(neureduce_out_dir / f"neureduce_vars_{num_vars}.txt", data)


def mbablast() -> None:
    """
    MBA-Blast: Unveiling and Simplifying Mixed Boolean-Arithmetic Obfuscation
    https://github.com/softsec-unh/MBA-Blast
    """
    mbablast_out_dir = OUT_DIR / "mbablast"
    if mbablast_out_dir.exists():
        print(f"MBA-Blast out dir exists at {mbablast_out_dir.as_posix()} -- skipping MBA-Blast")
        return

    vars_to_data: Dict[int, Set[str]] = defaultdict(set)
    for file_ in MBABLAST_PATH.glob("dataset*"):
        print(f"Processing {file_}")
        with open(file_, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if (line := clean_line(line)) is not None:
                    line, vars = unify_vars(line)
                    assert line.count(",") == 1, f"MBA-Blast {file_.as_posix()}: Found {line.count(',')} commas in line: {line}"
                    vars_to_data[len(vars)].add(line)
    # save data
    mbablast_out_dir.mkdir()
    for num_vars, data in vars_to_data.items():
        print(f"MBA-Blast: Saving {len(data)} for {num_vars} variables")
        save(mbablast_out_dir / f"mbablast_vars_{num_vars}.txt", data)


def mbaobfuscator() -> None:
    """
    Software Obfuscation with Non-Linear Mixed Boolean-Arithmetic Expressions
    https://github.com/nhpcc502/MBA-Obfuscator
    """
    mbaobfuscator_out_dir = OUT_DIR / "mbaobfuscator"
    if mbaobfuscator_out_dir.exists():
        print(f"MBA-Obfuscator out dir exists at {mbaobfuscator_out_dir.as_posix()} -- skipping MBA-Obfuscator")
        return

    vars_to_data: Dict[int, Set[str]] = defaultdict(set)
    for file_ in MBAOBFUSCATOR_PATH.glob("lMBA*"):
        print(f"Processing {file_}")
        print(f"Mba-Obfuscator: Processing {file_}")
        with open(file_, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if (line := clean_line(line)) is not None:
                    line, vars = unify_vars(line)
                    assert line.count(",") == 1, f"MBA-Obfuscator {file_.as_posix()}: " \
                                             f"Found {line.count(',')} commas in line: {line}"
                    vars_to_data[len(vars)].add(line)
    # save data
    mbaobfuscator_out_dir.mkdir()
    for num_vars, data in vars_to_data.items():
        print(f"MBA-Obfuscator: Saving {len(data)} for {num_vars} variables")
        save(mbaobfuscator_out_dir / f"mbaobfuscator_vars_{num_vars}.txt", data)


def loki() -> None:
    """
    Loki: Hardening Code Obfuscation Against Automated Attacks
    https://github.com/RUB-SysSec/loki
    """
    loki_out_dir = OUT_DIR / "loki"
    if loki_out_dir.exists():
        print(f"Loki out dir exists at {loki_out_dir.as_posix()} -- skipping Loki")
        return

    depths_to_data: Dict[str, Set[str]] = {}
    for file_ in LOKI_PATH.glob("*"):
        print(f"Processing {file_}")
        depth = file_.with_suffix("").name.split("depth")[1]
        depths_to_data[depth] = set()
        if file_.name.startswith("add"):
            baseline = "x + y"
        elif file_.name.startswith("sub"):
            baseline = "x - y"
        elif file_.name.startswith("and"):
            baseline = "x & y"
        elif file_.name.startswith("or"):
            baseline = "x | y"
        elif file_.name.startswith("xor"):
            baseline = "x ^ y"
        else:
            raise RuntimeError(f"Failed to match filename to operation: {file_.name}")
        with open(file_, "r", encoding="utf-8") as f:
            for line in f.readlines():
                depths_to_data[depth].add(line.strip() + f",{baseline}\n")
    # save data
    loki_out_dir.mkdir()
    for depth, data in depths_to_data.items():
        print(f"Loki: Saving {len(data)} MBAs for depth {depth}")
        save(loki_out_dir / f"loki_vars_2_depth_{depth}.txt", data)


def verify_ops(expr: str, vars: Set[str]) -> bool:
    """
    Filter out unsupported operations from simplified expression
    """
    allowed_chars = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", " "}
    compl_expr, simp_expr = expr.split(",", 1)
    diff = set(compl_expr)\
                        .difference(vars)\
                        .difference(KNOWN_OPS)\
                        .difference(allowed_chars)
    if diff:
        print(f"[!] Complex expression contains unsupported operation(s): {diff}")
        return False
    diff = set(simp_expr.strip())\
                        .difference(vars)\
                        .difference(KNOWN_OPS)\
                        .difference(allowed_chars)
    if diff:
        print(f"[!] Simplified expression contains unsupported operation(s): {diff}")
        return False
    return True


def unify_dataset() -> None:
    """
    Unify MBAs of all tools into a unified sets of MBAs (one set per variable number)
    """
    if not OUT_DIR.exists():
        print(f"Processed dataset at {OUT_DIR.as_posix()} does not exist")
        return
    if UNIFIED_OUT_DIR.exists():
        print(f"Unified out dir exists at {OUT_DIR.as_posix()} -- skipping unification")
        return

    vars_to_data: Dict[int, Set[str]] = defaultdict(set)
    for file_ in OUT_DIR.glob("*/*.txt"):
        with open(file_, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip(",True").rstrip(",False")
                assert line.count(",") == 1, f"Counted {line.count(',')} comma characters, expected 1: {line}"
                vars = get_vars(line)
                if not verify_ops(line, vars):
                    print(f"[!] Skipping due to unsupported ops: {line}")
                    continue
                assert vars == set(sort_variables(KNOWN_VARS)[:len(vars)]), \
                        f"Set mismatch: found {vars} but expected {KNOWN_VARS}"
                assert "x" in line, f"Expected variable x to be present: {line}"
                assert "y" in line, f"Expected variable y to be present: {line}"
                vars_to_data[len(vars)].add(line)

    UNIFIED_OUT_DIR.mkdir()
    for num_vars, data in vars_to_data.items():
        print(f"Unified: Saving {len(data)} for {num_vars} variables")
        save(UNIFIED_OUT_DIR / f"vars_{num_vars}.txt", data)


def split_by_operation(dataset: List[str]) -> Dict[int, List[str]]:
    """
    Given a dataset containing lines of format 'obfuscated_expr, simplified_expr',
    return a dict mapping number of operations to expressions with that number
    """
    exprs_by_ops = defaultdict(list)
    for expr in dataset:
        comp_expr, _ = expr.split(",", 1)
        num_ops = len([c for c in comp_expr if c in KNOWN_OPS])
        exprs_by_ops[num_ops].append(expr)
    return exprs_by_ops


def split_unified_set() -> None:
    """
    Split unified dataset according to the number of operations in the simplified expression
    """
    if not OUT_DIR.exists():
        print(f"Processed dataset at {OUT_DIR.as_posix()} does not exist")
        return
    if not UNIFIED_OUT_DIR.exists():
        print(f"Unified out dir does not exist at {UNIFIED_OUT_DIR.as_posix()} -- skipping splitting")
        return
    unified_ops_out_dir = OUT_DIR / "unified_by_ops"
    if unified_ops_out_dir.exists():
        print(
            f"unified_by_ops out dir exists at {unified_ops_out_dir.as_posix()} " \
            f"-- skipping splitting of unified data by ops"
        )
        return
    unified_ops_out_dir.mkdir()

    for unified_file in UNIFIED_OUT_DIR.glob("*.txt"):
        if "ops" in unified_file.name:
            continue
        print(f"Splitting dataset in {unified_file.as_posix()}")
        # load
        with open(unified_file, "r", encoding="utf-8") as f:
            dataset = f.readlines()
        # split by operations
        by_ops = split_by_operation(dataset)
        for num_ops, op_dataset in by_ops.items():
            # save
            out_file = unified_ops_out_dir / (unified_file.with_suffix("").name + f"_ops_{num_ops}.txt")
            print(f"Saving expressions with {num_ops} #ops to {out_file.name}")
            save(out_file, op_dataset)


def main(args: Namespace) -> None:
    """
    Process all tools and unify data finally
    """
    if args.parse:
        OUT_DIR.mkdir()
        loki()
        neureduce()
        mbablast()
        mbaobfuscator()
    if args.unify:
        unify_dataset()
    if args.split_by_ops:
        split_unified_set()


if __name__ == "__main__":
    parser = ArgumentParser(description="Unify MBA datasets")
    parser.add_argument("--parse", action="store_true", default=False, help="Parse tools and import their datasets")
    parser.add_argument("--unify", action="store_true", default=False, help="Unify imported datasets")
    parser.add_argument("--split-by-ops", dest="split_by_ops", action="store_true",
                        default=False, help="Split unified datasets according to the number of operations in " \
                        "the simplified expression.")
    args = parser.parse_args()
    if not any((args.parse, args.unify, args.split_by_ops)):
        parser.print_usage()
        print(f"No operation selected: Select one from --parse, --unify, or --split-by-ops")
        exit(1)
    main(args)
