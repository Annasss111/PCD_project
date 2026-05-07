import json
import csv
from pathlib import Path

instances = [
    "openstack-kolla-ansible-stable-2023.1-I9e0d656",
    "openstack-trove-stable-stein-I35e17af",
    "openstack-tripleo-ansible-stable-train-I821d674",
    "openstack-kolla-ansible-stable-queens-I2c1e991",
    "openstack-tripleo-ansible-stable-train-I9844ed1",
    "openstack-openstack-ansible-os_keystone-stable-ocata-I462e649",
    "openstack-kolla-ansible-stable-rocky-I10c82dc",
    "openstack-kolla-ansible-stable-queens-I10c82dc",
    "openstack-tripleo-ansible-stable-train-I9781007",
    "openstack-tripleo-validations-stable-victoria-I9869afd",
    "openstack-tripleo-validations-stable-ussuri-I9869afd",
    "openstack-tripleo-validations-stable-train-I9869afd",
    "openstack-tripleo-validations-stable-train-Ie45f8a6",
]

def guess_label(data):
    reason = data.get("failure_reason")
    patch = data.get("patch_applies")
    reg = data.get("regression_tests_pass")
    oracle_apply = data.get("oracle_test_patch_applies")
    oracle = data.get("oracle_tests_pass")

    if reason == "llm_file_edit_produced_empty_diff":
        return "llm_noop_empty_diff"
    if reason == "test_environment_failed":
        return "test_environment_failed"
    if reason == "clone_failed":
        return "clone_failed"
    if patch is True and reg is True and oracle_apply is True and oracle is True:
        return "full_success"
    if patch is True and reg is False and oracle_apply is True and oracle is True:
        return "oracle_pass_regression_fail"
    if patch is True and oracle_apply is True and oracle is False:
        return "needs_inspection_or_llm_failure"
    if reason:
        return reason
    return "unknown"

rows = []

for inst in instances:
    path = Path("onefile_runs") / inst / "eval_result.json"
    if not path.exists():
        rows.append({
            "instance_id": inst,
            "project": "",
            "target_stable_branch": "",
            "patch_applies": "",
            "regression_tests_pass": "",
            "oracle_test_patch_applies": "",
            "oracle_tests_pass": "",
            "failure_reason": "missing_eval_result",
            "label_guess": "missing_eval_result",
        })
        continue

    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    rows.append({
        "instance_id": inst,
        "project": data.get("project"),
        "target_stable_branch": data.get("target_stable_branch"),
        "patch_applies": data.get("patch_applies"),
        "regression_tests_pass": data.get("regression_tests_pass"),
        "oracle_test_patch_applies": data.get("oracle_test_patch_applies"),
        "oracle_tests_pass": data.get("oracle_tests_pass"),
        "failure_reason": data.get("failure_reason"),
        "label_guess": guess_label(data),
    })

with open("batch30_resume_partial_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Wrote batch30_resume_partial_summary.csv")
for row in rows:
    print(row["instance_id"], "=>", row["label_guess"])
