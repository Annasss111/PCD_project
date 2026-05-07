import argparse
import base64
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


CSV_PATH = "modified_backports_124_stable_test_check.csv"
DEFAULT_BASE_URL = "https://review.opendev.org/changes/"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct-fp16"

OUTPUT_DIR = Path("onefile_runs")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FILE_CHARS = 60000
MAX_FILE_DIFF_CHARS = 40000

OLD_BRANCHES_USE_PY38 = {
    "stable/ocata",
    "stable/pike",
    "stable/queens",
    "stable/rocky",
    "stable/stein",
    "stable/train",
    "stable/ussuri",
    "stable/victoria",
    "stable/wallaby",
}


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_json_list(value):
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def extract_change_number(gerrit_url):
    match = re.search(r"/\+/(\d+)", str(gerrit_url))
    if not match:
        raise ValueError(f"Could not extract change number from URL: {gerrit_url}")
    return match.group(1)


def strip_gerrit_prefix(text):
    if text.startswith(")]}'"):
        return text.split("\n", 1)[1]
    return text


def gerrit_get_json(path):
    url = f"{DEFAULT_BASE_URL}{path}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return json.loads(strip_gerrit_prefix(response.text))


def get_stable_commit_info(stable_backport_url):
    change_number = extract_change_number(stable_backport_url)
    data = gerrit_get_json(f"{quote(change_number, safe='')}/revisions/current/commit")

    stable_commit = data["commit"]
    parents = data.get("parents", [])
    if not parents:
        raise ValueError(f"No parent commit found for {stable_backport_url}")

    base_commit = parents[0]["commit"]
    return stable_commit, base_commit


def decode_gerrit_patch_response(text):
    stripped = text.strip()

    if "diff --git" in stripped or stripped.startswith("From "):
        return stripped

    decoded = base64.b64decode(stripped, validate=True)
    return decoded.decode("utf-8", errors="replace")


def fetch_patch(change_url):
    change_number = extract_change_number(change_url)
    url = f"{DEFAULT_BASE_URL}{quote(change_number, safe='')}/revisions/current/patch?download"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    return decode_gerrit_patch_response(response.text)


def parse_patch_paths(patch_text):
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b_path = parts[3]
                if b_path.startswith("b/"):
                    path = b_path[2:]
                    if path not in paths and path != "/dev/null":
                        paths.append(path)
    return paths


def extract_file_diff(patch_text, target_path):
    lines = patch_text.splitlines()
    chunks = []
    in_target = False

    for line in lines:
        if line.startswith("diff --git "):
            if in_target:
                break

            parts = line.split()
            if len(parts) >= 4:
                b_path = parts[3]
                path = b_path[2:] if b_path.startswith("b/") else b_path
                in_target = path == target_path
                if in_target:
                    chunks.append(line)
            continue

        if in_target:
            chunks.append(line)

    return "\n".join(chunks).strip()


def is_test_path(path):
    p = path.lower().replace("\\", "/")
    parts = p.split("/")
    filename = parts[-1]

    test_dirs = {
        "test",
        "tests",
        "unit_test",
        "unit_tests",
        "functional",
        "integration",
        "tempest",
        "molecule",
    }

    if any(part in test_dirs for part in parts):
        return True

    if filename.startswith("test_") and filename.endswith(".py"):
        return True

    return False


def is_python_test_file(path):
    p = path.lower().replace("\\", "/")
    return p.endswith(".py") and "/molecule/" not in p


def is_unit_test_file(path):
    p = path.lower().replace("\\", "/")
    return p.endswith(".py") and "/tests/unit/" in p


def choose_docker_image(target_stable_branch):
    if target_stable_branch in OLD_BRANCHES_USE_PY38:
        return "backport-eval-py38"
    return "backport-eval"


def select_candidates(df, limit, include_functional, instance_id):
    if instance_id:
        selected = df[df["instance_id"] == instance_id].copy()
        if selected.empty:
            raise ValueError(f"Instance not found: {instance_id}")
        return selected

    rows = []
    for _, row in df.iterrows():
        if not as_bool(row.get("stable_has_real_tests_v2")):
            continue

        test_files = parse_json_list(row.get("stable_real_test_files_v2"))
        python_tests = [f for f in test_files if is_python_test_file(f)]

        if include_functional:
            selected_tests = python_tests
        else:
            selected_tests = [f for f in python_tests if is_unit_test_file(f)]

        if not selected_tests:
            continue

        row = row.copy()
        row["_selected_test_files"] = selected_tests
        rows.append(row)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No suitable Python test candidates found.")

    candidates["stable_diff_size"] = (
        candidates["stable_lines_added"].fillna(0)
        + candidates["stable_lines_removed"].fillna(0)
    )
    candidates = candidates.sort_values("stable_diff_size", ascending=True)
    return candidates.head(limit)


def call_ollama(model, prompt, timeout_seconds):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()["response"]


def build_one_file_prompt(row, file_path, file_diff, stable_content):
    if len(file_diff) > MAX_FILE_DIFF_CHARS:
        file_diff = file_diff[:MAX_FILE_DIFF_CHARS] + "\n[FILE DIFF TRUNCATED]\n"

    if stable_content is None:
        stable_content = "[THIS FILE DOES NOT EXIST IN THE STABLE BASE]"
    elif len(stable_content) > MAX_FILE_CHARS:
        stable_content = stable_content[:MAX_FILE_CHARS] + "\n[STABLE FILE TRUNCATED]\n"

    return f"""
You are editing exactly one file for an OpenStack stable-branch backport.

Project: {row["project"]}
Target branch: {row["target_stable_branch"]}
Instance: {row["instance_id"]}
File to edit: {file_path}

Your job:
Produce the complete final content of this one file for the stable branch.

Important output rules:
- Do not output a git diff.
- Do not write explanations.
- Do not write a summary.
- Do not use Markdown.
- Do not use code fences.
- Return raw final file content only.
- Do not include diff markers.
- No line should start with "+" or "-" unless that character is truly part of the final source file.
- Output exactly one of these formats.

If the file should be changed, output:

===BEGIN FILE CONTENT===
complete final file content here
===END FILE CONTENT===

If the file should not be changed, output exactly:

===NO CHANGE===

Master-side diff for this file:
----- BEGIN FILE DIFF -----
{file_diff}
----- END FILE DIFF -----

Stable-base content of this file:
----- BEGIN STABLE FILE -----
{stable_content}
----- END STABLE FILE -----
""".strip()


def clean_full_file_content(content):
    """
    The model sometimes copies diff-style leading markers into a full file.
    Examples:
        +    migrate = False
        ++   migrate = False

    Since this script writes full files, those markers would become invalid source.
    This cleaner removes the accidental markers and repairs common indentation loss.
    """
    lines = content.splitlines()
    marker_lines = sum(
        1 for line in lines
        if re.match(r"^[+-]+\s", line)
    )

    if marker_lines >= 2:
        cleaned = []
        for line in lines:
            # Drop normal diff header lines if the model included them.
            if line.startswith("+++") or line.startswith("---"):
                continue

            # Remove one or more accidental leading '+' markers.
            plus_match = re.match(r"^(\++)(\s*)(.*)$", line)
            if plus_match:
                spaces = plus_match.group(2)
                rest = plus_match.group(3)

                # If removing "++" leaves 3/7/11 spaces, restore to 4/8/12.
                if len(spaces) % 4 == 3:
                    spaces = " " + spaces

                line = spaces + rest

            # Drop likely removed diff lines from full-file content.
            elif re.match(r"^-+\s", line):
                continue

            cleaned.append(line)

        return "\n".join(cleaned).rstrip("\n") + "\n"

    return content.rstrip("\n") + "\n"



def parse_one_file_response(text):
    text = text.strip()

    if "===NO CHANGE===" in text:
        return None, "no_change"

    match = re.search(
        r"===BEGIN FILE CONTENT===\s*\n(.*?)\n===END FILE CONTENT===",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return None, "parse_failed"

    content = clean_full_file_content(match.group(1))
    return content, "parsed"


PREPARE_STABLE_FILES_SCRIPT = r"""
import json
import subprocess
from pathlib import Path


CASE_DIR = Path("/case")
WORK_DIR = Path("/work/repo")


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout[-4000:], flush=True)
    return result


def main():
    metadata = json.loads((CASE_DIR / "metadata.json").read_text())
    repo_url = metadata["repo_url"]
    base_commit = metadata["base_commit"]
    target_code_files = metadata["target_code_files"]

    clone = run(["git", "clone", repo_url, str(WORK_DIR)])
    if clone.returncode != 0:
        raise SystemExit(clone.returncode)

    checkout = run(["git", "checkout", base_commit], cwd=WORK_DIR)
    if checkout.returncode != 0:
        raise SystemExit(checkout.returncode)

    stable_files = {}
    for path in target_code_files:
        show = subprocess.run(
            ["git", "show", f"{base_commit}:{path}"],
            cwd=WORK_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if show.returncode == 0:
            stable_files[path] = show.stdout
        else:
            stable_files[path] = None

    (CASE_DIR / "stable_files.json").write_text(
        json.dumps(stable_files, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
"""


CONTAINER_EVAL_SCRIPT = r"""
import json
import re
import subprocess
from pathlib import Path


CASE_DIR = Path("/case")
WORK_DIR = Path("/work/repo")


def run(cmd, cwd=None, timeout=1800, output_file=None):
    print("+", " ".join(cmd), flush=True)

    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )

    if output_file is not None:
        Path(output_file).write_text(result.stdout)

    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "output": result.stdout[-12000:],
    }


def test_file_exists(path):
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=WORK_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def run_tox_tests(test_files, tox_env):
    modules = [
        path[:-3].replace("/", ".")
        for path in test_files
        if path.endswith(".py")
    ]

    if not modules:
        return {
            "skipped": True,
            "reason": "No Python test modules selected",
        }

    metadata = json.loads((CASE_DIR / "metadata.json").read_text())
    project = metadata.get("project")

    # Kolla's stestr discovery can import unrelated tests and hit Ansible
    # dependency conflicts. For selected unit modules, build the tox env,
    # then run only the requested module(s) directly with unittest.
    if project == "openstack/kolla-ansible":
        setup = run(["tox", "-e", tox_env, "--notest"], cwd=WORK_DIR, timeout=7200)
        if setup["returncode"] != 0:
            setup["stage"] = "tox_setup"
            return setup

        python_bin = WORK_DIR / ".tox" / tox_env / "bin" / "python"
        cmd = [str(python_bin), "-m", "unittest"] + modules
        result = run(cmd, cwd=WORK_DIR, timeout=7200)
        result["stage"] = "direct_unittest_after_tox_setup"
        return result

    cmd = ["tox", "-e", tox_env, "--"] + modules
    return run(cmd, cwd=WORK_DIR, timeout=7200)


def patch_tox_ini_for_env_compat(project=None, target_branch=None):
    # Compatibility fixes for old OpenStack stable branches.
    # This is evaluation-only; it must not be included in model_output.patch.
    tox_ini = WORK_DIR / "tox.ini"
    if not tox_ini.exists():
        return {"skipped": True, "reason": "tox.ini not found"}

    text = tox_ini.read_text(encoding="utf-8")
    marker = "# backport-eval env compatibility pin"
    if marker in text:
        return {"changed": False, "reason": "already patched"}

    deps_to_add = [
        marker,
        "setuptools<58",
        "pip<24",
        "wheel",
    ]

    # Older Kolla branches can pull ansible-lint/ansible-core combinations
    # that break imports during test discovery.
    if project == "openstack/kolla-ansible":
        deps_to_add.append("ansible-lint<5")

    lines = text.splitlines()
    out = []
    in_testenv = False
    inserted = False

    def add_deps(target):
        for dep in deps_to_add:
            target.append("    " + dep)

    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)

        if line.strip() == "[testenv]":
            in_testenv = True
            i += 1
            continue

        if in_testenv and line.startswith("[") and line.strip() != "[testenv]":
            if not inserted:
                out.insert(len(out) - 1, "deps =")
                insert_at = len(out) - 1
                for dep in deps_to_add:
                    out.insert(insert_at, "    " + dep)
                    insert_at += 1
                inserted = True
            in_testenv = False

        if in_testenv and re.match(r"^\s*deps\s*=", line):
            add_deps(out)
            inserted = True
            i += 1
            continue

        i += 1

    if in_testenv and not inserted:
        out.append("deps =")
        add_deps(out)
        inserted = True

    if not inserted:
        out.append("")
        out.append("[testenv]")
        out.append("deps =")
        add_deps(out)

    tox_ini.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"changed": True, "deps_added": deps_to_add}


def is_environment_failure(output):
    patterns = [
        "canonicalize_version() got an unexpected keyword argument",
        "metadata-generation-failed",
        "No module named 'pkg_resources'",
        "ModuleNotFoundError: No module named 'pkg_resources'",
        "AnsibleCollectionLoader",
        "get_script_header",
    ]
    return any(pattern in output for pattern in patterns)


def write_llm_files(llm_files):
    for rel_path, content in llm_files.items():
        path = WORK_DIR / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def run_python_syntax_check(llm_files):
    py_files = [path for path in llm_files if path.endswith(".py")]
    if not py_files:
        return {
            "skipped": True,
            "reason": "No edited Python files",
            "returncode": 0,
        }

    cmd = ["python3", "-m", "py_compile"] + py_files
    return run(cmd, cwd=WORK_DIR, timeout=1800)


def main():
    metadata = json.loads((CASE_DIR / "metadata.json").read_text())
    llm_files = json.loads((CASE_DIR / "llm_files.json").read_text())

    repo_url = metadata["repo_url"]
    base_commit = metadata["base_commit"]
    stable_commit = metadata["stable_commit"]
    test_files = metadata["selected_test_files"]
    tox_env = metadata["tox_env"]

    result = {
        "instance_id": metadata["instance_id"],
        "project": metadata["project"],
        "base_commit": base_commit,
        "stable_commit": stable_commit,
        "tox_env": tox_env,
        "selected_test_files": test_files,
    }

    clone = run(["git", "clone", repo_url, str(WORK_DIR)], timeout=1800)
    result["clone"] = clone
    if clone["returncode"] != 0:
        result["failure_reason"] = "clone_failed"
        (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
        return

    checkout = run(["git", "checkout", base_commit], cwd=WORK_DIR)
    result["checkout_base"] = checkout
    if checkout["returncode"] != 0:
        result["failure_reason"] = "checkout_base_failed"
        (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
        return

    existing_tests = [path for path in test_files if test_file_exists(path)]
    result["existing_tests_before_oracle"] = existing_tests

    write_llm_files(llm_files)

    diff_result = run(
        ["git", "diff", "--binary"],
        cwd=WORK_DIR,
        timeout=1800,
        output_file=CASE_DIR / "model_output.patch",
    )
    result["model_patch_generate"] = diff_result

    model_patch_text = (CASE_DIR / "model_output.patch").read_text()
    result["model_patch_nonempty"] = bool(model_patch_text.strip())

    if not model_patch_text.strip():
        result["patch_applies"] = False
        result["failure_reason"] = "llm_file_edit_produced_empty_diff"
        (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
        return

    result["patch_applies"] = True

    syntax_check = run_python_syntax_check(llm_files)
    result["python_syntax_check"] = syntax_check

    if syntax_check.get("returncode") != 0:
        result["regression_tests_pass"] = None
        result["oracle_test_patch_applies"] = None
        result["oracle_tests_pass"] = None
        result["failure_reason"] = "python_syntax_error"
        (CASE_DIR / "python_syntax_error.log").write_text(
            syntax_check.get("output", ""),
            encoding="utf-8",
        )
        (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
        return

    env_patch = patch_tox_ini_for_env_compat(
        project=metadata.get("project"),
        target_branch=metadata.get("target_stable_branch"),
    )
    result["env_compat_patch"] = env_patch

    if existing_tests:
        regression = run_tox_tests(existing_tests, tox_env)
        result["regression_tests"] = regression
        result["regression_tests_pass"] = regression.get("returncode") == 0

        if regression.get("returncode") != 0 and is_environment_failure(regression.get("output", "")):
            result["oracle_test_patch_applies"] = None
            result["oracle_tests_pass"] = None
            result["failure_reason"] = "test_environment_failed"
            (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
            return
    else:
        result["regression_tests"] = {
            "skipped": True,
            "reason": "Selected tests do not exist before oracle test patch",
        }
        result["regression_tests_pass"] = None

    oracle_patch_path = CASE_DIR / "oracle_test_only.patch"
    oracle_diff_cmd = ["git", "diff", base_commit, stable_commit, "--"] + test_files
    oracle_diff = run(
        oracle_diff_cmd,
        cwd=WORK_DIR,
        timeout=1800,
        output_file=oracle_patch_path,
    )
    result["oracle_test_patch_generate"] = oracle_diff

    if oracle_diff["returncode"] != 0:
        result["failure_reason"] = "oracle_test_patch_generation_failed"
        (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
        return

    oracle_patch_text = oracle_patch_path.read_text()
    result["oracle_test_patch_nonempty"] = bool(oracle_patch_text.strip())

    if not oracle_patch_text.strip():
        result["oracle_test_patch_applies"] = None
        result["oracle_tests_pass"] = None
        result["failure_reason"] = "oracle_test_patch_empty"
        (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
        return

    oracle_check = run(
        ["git", "apply", "--check", str(oracle_patch_path)],
        cwd=WORK_DIR,
    )
    result["oracle_test_patch_check"] = oracle_check

    if oracle_check["returncode"] != 0:
        result["oracle_test_patch_applies"] = False
        result["failure_reason"] = "oracle_test_patch_does_not_apply_after_llm"
        (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
        return

    oracle_apply = run(
        ["git", "apply", str(oracle_patch_path)],
        cwd=WORK_DIR,
    )
    result["oracle_test_patch_apply"] = oracle_apply
    result["oracle_test_patch_applies"] = oracle_apply["returncode"] == 0

    if oracle_apply["returncode"] != 0:
        result["failure_reason"] = "oracle_test_patch_apply_failed"
        (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))
        return

    oracle_tests = run_tox_tests(test_files, tox_env)
    result["oracle_tests"] = oracle_tests
    result["oracle_tests_pass"] = oracle_tests.get("returncode") == 0

    if result["oracle_tests_pass"]:
        result["failure_reason"] = None
    elif is_environment_failure(oracle_tests.get("output", "")):
        result["failure_reason"] = "test_environment_failed"
    else:
        result["failure_reason"] = "oracle_tests_failed"

    (CASE_DIR / "eval_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
"""


def run_container_script(run_dir, docker_image, script_text, script_name):
    script_path = run_dir / script_name
    script_path.write_text(script_text, encoding="utf-8")

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{run_dir.resolve()}:/case",
        docker_image,
        "python3",
        f"/case/{script_name}",
    ]

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10800,
    )

    (run_dir / f"{script_name}.log").write_text(result.stdout, encoding="utf-8")
    return result.returncode


def safe_name(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path)


def write_results_csv(results):
    output = OUTPUT_DIR / "results.csv"
    fields = [
        "instance_id",
        "project",
        "target_stable_branch",
        "docker_image",
        "model",
        "target_files",
        "llm_file_blocks_parsed",
        "patch_applies",
        "regression_tests_pass",
        "oracle_test_patch_applies",
        "oracle_tests_pass",
        "failure_reason",
        "run_dir",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})

    print(f"Results written to: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tox-env", default="py3")
    parser.add_argument("--include-functional", action="store_true")
    parser.add_argument("--skip-llm-if-exists", action="store_true")
    parser.add_argument("--ollama-timeout", type=int, default=3600)
    parser.add_argument("--python-only-targets", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    selected = select_candidates(
        df=df,
        limit=args.limit,
        include_functional=args.include_functional,
        instance_id=args.instance_id,
    )

    print(f"Selected {len(selected)} instances")
    summary_results = []

    for _, row in selected.iterrows():
        instance_id = row["instance_id"]
        run_dir = OUTPUT_DIR / instance_id
        run_dir.mkdir(parents=True, exist_ok=True)

        target_branch = row["target_stable_branch"]
        docker_image = choose_docker_image(target_branch)
        repo_url = f"https://opendev.org/{row['project']}.git"

        print()
        print("=" * 80)
        print(f"Instance: {instance_id}")
        print(f"Project: {row['project']}")
        print(f"Stable branch: {target_branch}")
        print(f"Docker image: {docker_image}")

        selected_test_files = row.get("_selected_test_files")
        if not isinstance(selected_test_files, list):
            selected_test_files = [
                f for f in parse_json_list(row.get("stable_real_test_files_v2"))
                if is_python_test_file(f)
            ]

        print("Selected tests:")
        for path in selected_test_files:
            print(" -", path)

        print("Fetching patches and stable commit info...")
        master_patch = fetch_patch(row["master_patch_url"])
        stable_patch = fetch_patch(row["stable_backport_url"])
        stable_commit, base_commit = get_stable_commit_info(row["stable_backport_url"])

        master_paths = parse_patch_paths(master_patch)
        target_code_files = [p for p in master_paths if not is_test_path(p)]
        if args.python_only_targets:
            target_code_files = [p for p in target_code_files if p.endswith(".py")]

        if not target_code_files:
            summary = {
                "instance_id": instance_id,
                "project": row["project"],
                "target_stable_branch": target_branch,
                "docker_image": docker_image,
                "model": args.model,
                "target_files": "",
                "llm_file_blocks_parsed": 0,
                "patch_applies": None,
                "regression_tests_pass": None,
                "oracle_test_patch_applies": None,
                "oracle_tests_pass": None,
                "failure_reason": "no_non_test_target_files_found",
                "run_dir": str(run_dir),
            }
            summary_results.append(summary)
            print(json.dumps(summary, indent=2))
            continue

        metadata = {
            "instance_id": instance_id,
            "project": row["project"],
            "repo_url": repo_url,
            "target_stable_branch": target_branch,
            "base_commit": base_commit,
            "stable_commit": stable_commit,
            "master_patch_url": row["master_patch_url"],
            "stable_backport_url": row["stable_backport_url"],
            "model": args.model,
            "docker_image": docker_image,
            "selected_test_files": selected_test_files,
            "target_code_files": target_code_files,
            "tox_env": args.tox_env,
        }

        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        (run_dir / "master.patch").write_text(master_patch, encoding="utf-8")
        (run_dir / "ground_truth_stable_from_gerrit.patch").write_text(
            stable_patch,
            encoding="utf-8",
        )

        print("Target code files:")
        for path in target_code_files:
            print(" -", path)

        print("Extracting stable file contents...")
        prep_code = run_container_script(
            run_dir=run_dir,
            docker_image=docker_image,
            script_text=PREPARE_STABLE_FILES_SCRIPT,
            script_name="prepare_stable_files.py",
        )

        if prep_code != 0 or not (run_dir / "stable_files.json").exists():
            summary = {
                "instance_id": instance_id,
                "project": row["project"],
                "target_stable_branch": target_branch,
                "docker_image": docker_image,
                "model": args.model,
                "target_files": "|".join(target_code_files),
                "llm_file_blocks_parsed": 0,
                "patch_applies": None,
                "regression_tests_pass": None,
                "oracle_test_patch_applies": None,
                "oracle_tests_pass": None,
                "failure_reason": "stable_file_extraction_failed",
                "run_dir": str(run_dir),
            }
            summary_results.append(summary)
            print(json.dumps(summary, indent=2))
            continue

        stable_files = json.loads(
            (run_dir / "stable_files.json").read_text(encoding="utf-8")
        )

        llm_files_path = run_dir / "llm_files.json"

        if args.skip_llm_if_exists and llm_files_path.exists():
            print("Using existing llm_files.json")
            llm_files = json.loads(llm_files_path.read_text(encoding="utf-8"))
            file_statuses = {}
        else:
            llm_files = {}
            file_statuses = {}

            for file_path in target_code_files:
                print(f"Calling Ollama for one file: {file_path}")
                file_diff = extract_file_diff(master_patch, file_path)
                stable_content = stable_files.get(file_path)

                prompt = build_one_file_prompt(
                    row=row,
                    file_path=file_path,
                    file_diff=file_diff,
                    stable_content=stable_content,
                )

                prompt_path = run_dir / f"prompt_{safe_name(file_path)}.txt"
                raw_path = run_dir / f"raw_model_output_{safe_name(file_path)}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")

                try:
                    raw_output = call_ollama(
                        model=args.model,
                        prompt=prompt,
                        timeout_seconds=args.ollama_timeout,
                    )
                except Exception as e:
                    file_statuses[file_path] = f"ollama_error: {e}"
                    continue

                raw_path.write_text(raw_output, encoding="utf-8")

                content, status = parse_one_file_response(raw_output)
                file_statuses[file_path] = status

                if status == "parsed":
                    llm_files[file_path] = content

                time.sleep(1)

            llm_files_path.write_text(
                json.dumps(llm_files, indent=2),
                encoding="utf-8",
            )
            (run_dir / "file_statuses.json").write_text(
                json.dumps(file_statuses, indent=2),
                encoding="utf-8",
            )

        if not llm_files:
            summary = {
                "instance_id": instance_id,
                "project": row["project"],
                "target_stable_branch": target_branch,
                "docker_image": docker_image,
                "model": args.model,
                "target_files": "|".join(target_code_files),
                "llm_file_blocks_parsed": 0,
                "patch_applies": None,
                "regression_tests_pass": None,
                "oracle_test_patch_applies": None,
                "oracle_tests_pass": None,
                "failure_reason": "llm_output_no_file_blocks",
                "run_dir": str(run_dir),
            }
            (run_dir / "eval_result.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            summary_results.append(summary)
            print(json.dumps(summary, indent=2))
            continue

        print(f"Parsed {len(llm_files)} edited file(s)")
        print("Running Docker evaluation...")
        docker_code = run_container_script(
            run_dir=run_dir,
            docker_image=docker_image,
            script_text=CONTAINER_EVAL_SCRIPT,
            script_name="container_eval_onefile.py",
        )

        eval_path = run_dir / "eval_result.json"
        if eval_path.exists():
            eval_result = json.loads(eval_path.read_text(encoding="utf-8"))
        else:
            eval_result = {
                "failure_reason": "docker_failed_without_eval_result",
                "docker_returncode": docker_code,
            }

        summary = {
            "instance_id": instance_id,
            "project": row["project"],
            "target_stable_branch": target_branch,
            "docker_image": docker_image,
            "model": args.model,
            "target_files": "|".join(target_code_files),
            "llm_file_blocks_parsed": len(llm_files),
            "patch_applies": eval_result.get("patch_applies"),
            "regression_tests_pass": eval_result.get("regression_tests_pass"),
            "oracle_test_patch_applies": eval_result.get("oracle_test_patch_applies"),
            "oracle_tests_pass": eval_result.get("oracle_tests_pass"),
            "failure_reason": eval_result.get("failure_reason"),
            "run_dir": str(run_dir),
        }

        summary_results.append(summary)
        print("Summary:")
        print(json.dumps(summary, indent=2))

    write_results_csv(summary_results)


if __name__ == "__main__":
    main()
