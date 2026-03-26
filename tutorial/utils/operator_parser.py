import ast
import json
import os
import argparse
from typing import Any, Dict, List


def _node_to_name(node: ast.AST) -> str:
    """Return a dotted name for Name/Attribute nodes, fall back to unparse for others."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    try:
        # Python 3.9+ has ast.unparse
        return ast.unparse(node)
    except Exception:
        return str(type(node))


def _expr_to_value(node: ast.AST) -> Any:
    """Try to evaluate simple literal expressions, otherwise return source-like string."""
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return str(node)


def parse_code_to_metadata(code: str) -> Dict[str, Any]:
    """Parse Python code string and extract functions, decorators and kernel-launch relations.

    Returns a dict with keys: `functions` (list), `kernels` (list of names), `wrappers` (list)
    Each function entry contains: name, args, defaults, decorators, called_kernels
    """
    tree = ast.parse(code)
    functions: List[Dict[str, Any]] = []
    kernel_names = []

    # collect all function defs first
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_entry: Dict[str, Any] = {
                "name": node.name,
                "args": [],
                "defaults": [],
                "decorators": [],
                "called_kernels": [],
            }

            # args
            for arg in node.args.args:
                func_entry["args"].append(arg.arg)

            # defaults
            for d in node.args.defaults:
                func_entry["defaults"].append(_expr_to_value(d))

            # decorators
            decs = []
            for d in node.decorator_list:
                decs.append(_node_to_name(d))
            func_entry["decorators"] = decs

            # identify kernels by decorator including 'triton'
            if any("triton" in dec for dec in decs):
                kernel_names.append(node.name)

            # find kernel launches inside function body: calls of form kernel[...]()
            called = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Subscript):
                    # kernel[(...)](...) style
                    kernel_name = _node_to_name(sub.func.value)
                    called.add(kernel_name)
                elif isinstance(sub, ast.Call) and isinstance(sub.func, (ast.Name, ast.Attribute)):
                    # sometimes kernel may be called directly
                    name = _node_to_name(sub.func)
                    if name in kernel_names:
                        called.add(name)

            func_entry["called_kernels"] = sorted(list(called))
            functions.append(func_entry)

    # mark wrappers as functions that call kernels
    wrappers = [f for f in functions if f["called_kernels"]]

    return {
        "functions": functions,
        "kernels": sorted(kernel_names),
        "wrappers": wrappers,
        "raw_len": len(code),
    }


def update_operator_dict(file_name: str,
                         instructions_json: str = "/Users/dongshouyang/Downloads/GEAK/tutorial/data/instructions.json",
                         out_json: str = "/Users/dongshouyang/Downloads/GEAK/tutorial/data/operators.json") -> Dict[str, Any]:
    """Load the instructions file, find the entry for `file_name`, parse the code, and update operators.json.

    Returns the written operator dict for the file.
    """
    if not os.path.exists(instructions_json):
        raise FileNotFoundError(f"instructions json not found: {instructions_json}")

    with open(instructions_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    entry = None
    # match by exact filename or by basename
    for e in entries:
        if e.get("file") == file_name or os.path.basename(e.get("file", "")) == file_name:
            entry = e
            break

    if entry is None:
        raise ValueError(f"No instruction entry found for file '{file_name}' in {instructions_json}")

    code = entry.get("output", "")
    metadata = parse_code_to_metadata(code)

    # construct operator dict record
    record = {
        "file": entry.get("file"),
        "difficulty": entry.get("difficulty"),
        "instruction": entry.get("instruction"),
        "metadata": metadata,
    }

    # load existing out json
    ops = {}
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                ops = json.load(f)
        except Exception:
            ops = {}

    ops[record["file"]] = record

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ops, f, indent=2, ensure_ascii=False)

    return record


def _cli_main():
    p = argparse.ArgumentParser(description="Parse instruction entry and update operators.json")
    p.add_argument("--file", "-f", required=True, help="file name in instructions.json (e.g. add_example.py)")
    p.add_argument("--instructions", default="/Users/dongshouyang/Downloads/GEAK/tutorial/data/instructions.json")
    p.add_argument("--out", default="/Users/dongshouyang/Downloads/GEAK/tutorial/data/operators.json")
    args = p.parse_args()

    rec = update_operator_dict(args.file, instructions_json=args.instructions, out_json=args.out)
    print("Updated operator entry for:", rec["file"])


def parse_kernel_file(kernel_path: str) -> str:
    """Read a kernel file and return the code before the separator line of hashes.

    This function prefers to split the file BEFORE a line that defines the
    `hash_line` variable (e.g. `hash_line = "#" * 146`). If that assignment
    is not present, it falls back to finding a line composed solely of `#`
    characters. If no separator is found, return the whole file.
    """
    if not os.path.exists(kernel_path):
        raise FileNotFoundError(kernel_path)
    with open(kernel_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sep_idx_assign = None
    sep_idx_hashline = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # prefer the assignment like: hash_line = "#" * 146
        if "hash_line" in stripped and "#" in stripped and "=" in stripped:
            sep_idx_assign = i
            break
        # fallback: a line composed only of # characters
        if stripped and set(stripped) == {"#"}:
            sep_idx_hashline = i

    if sep_idx_assign is not None:
        return "".join(lines[:sep_idx_assign])
    if sep_idx_hashline is not None:
        return "".join(lines[:sep_idx_hashline])
    return "".join(lines)


def generate_instruction_from_kernel(kernel_file: str,
                                     kernel_dir: str = "/Users/dongshouyang/Downloads/GEAK/tutorial/data/kernels",
                                     out_instructions: str = "/Users/dongshouyang/Downloads/GEAK/tutorial/data/generated_instructions.json",
                                     difficulty: str = "3") -> Dict[str, Any]:
    """Create an instructions.json-style entry from a kernel file.

    The entry will include `instruction` (auto message), empty `input`, `output` containing the code
    (only up to the separator), `file` with basename, and `difficulty`.
    """
    # Resolve the kernel file path robustly:
    # 1) absolute path -> use it
    # 2) path exists as given (relative to cwd) -> use it
    # 3) exists under kernel_dir -> join and use
    # 4) basename under kernel_dir -> join and use
    # Otherwise raise FileNotFoundError
    if os.path.isabs(kernel_file):
        path = kernel_file
    elif os.path.exists(kernel_file):
        path = kernel_file
    else:
        candidate = os.path.join(kernel_dir, kernel_file)
        if os.path.exists(candidate):
            path = candidate
        else:
            candidate2 = os.path.join(kernel_dir, os.path.basename(kernel_file))
            if os.path.exists(candidate2):
                path = candidate2
            else:
                raise FileNotFoundError(f"Kernel file not found: {kernel_file}")

    code = parse_kernel_file(path)
    entry = {
        "instruction": f"Auto-generated from kernel file {os.path.basename(path)}. Implement the Triton operator as in the code.",
        "input": "",
        "output": code,
        "file": os.path.basename(path),
        "difficulty": difficulty,
    }

    # write or append to output json
    data = []
    if os.path.exists(out_instructions):
        try:
            with open(out_instructions, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []

    data.append(entry)
    os.makedirs(os.path.dirname(out_instructions), exist_ok=True)
    with open(out_instructions, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return entry


def _cli_main_extra():
    p = argparse.ArgumentParser(description="Generate instruction entry from a kernel file")
    p.add_argument("--kernel-file", "-k", required=False, help="path or basename of kernel file under tutorial/data/kernels")
    p.add_argument("--out", default="/Users/dongshouyang/Downloads/GEAK/tutorial/data/generated_instructions.json")
    p.add_argument("--difficulty", default="3")
    args = p.parse_args()
    if not args.kernel_file:
        print("Please pass --kernel-file to generate an instruction entry from a kernel file.")
        return
    rec = generate_instruction_from_kernel(args.kernel_file, out_instructions=args.out, difficulty=args.difficulty)
    print("Generated instruction for:", rec["file"]) 


if __name__ == "__main__":
    # If called directly with --kernel-file use the extra CLI, otherwise keep previous behavior
    import sys
    if "--kernel-file" in sys.argv or "-k" in sys.argv:
        _cli_main_extra()
    else:
        _cli_main()

