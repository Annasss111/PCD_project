import csv

rows = [
    {
        "instance_id": "openstack-neutron-stable-wallaby-I29a3910",
        "project": "openstack/neutron",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "full_success",
        "notes": "Passed regression and oracle tests",
    },
    {
        "instance_id": "openstack-kolla-ansible-stable-wallaby-I7e9d5c9",
        "project": "openstack/kolla-ansible",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "oracle_pass_regression_fail",
        "notes": "Oracle tests passed but old regression tests failed",
    },
    {
        "instance_id": "openstack-kolla-ansible-stable-victoria-I7e9d5c9",
        "project": "openstack/kolla-ansible",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "oracle_pass_regression_fail",
        "notes": "Oracle tests passed after Kolla environment fix",
    },
    {
        "instance_id": "openstack-kolla-ansible-stable-ussuri-I7e9d5c9",
        "project": "openstack/kolla-ansible",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "oracle_pass_regression_fail",
        "notes": "Oracle tests passed after Kolla environment fix",
    },
    {
        "instance_id": "openstack-kolla-ansible-stable-ussuri-Ia8623be",
        "project": "openstack/kolla-ansible",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "llm_noop_empty_diff",
        "notes": "LLM produced no effective code change",
    },
    {
        "instance_id": "openstack-kolla-ansible-stable-train-Ia8623be",
        "project": "openstack/kolla-ansible",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "llm_noop_empty_diff",
        "notes": "LLM produced no effective code change",
    },
    {
        "instance_id": "openstack-ironic-stable-2024.1-I7fac5c6",
        "project": "openstack/ironic",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "llm_test_failure",
        "notes": "Generated code failed real tests",
    },
    {
        "instance_id": "openstack-ironic-stable-2023.1-I7fac5c6",
        "project": "openstack/ironic",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "llm_test_failure",
        "notes": "Generated code failed real tests",
    },
    {
        "instance_id": "openstack-ironic-stable-2023.2-I7fac5c6",
        "project": "openstack/ironic",
        "model": "qwen2.5-coder:7b-instruct-fp16",
        "final_label": "llm_test_failure",
        "notes": "Generated code failed real tests",
    },
]

with open("confirmed_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["instance_id", "project", "model", "final_label", "notes"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("Wrote confirmed_results.csv with", len(rows), "confirmed rows")
