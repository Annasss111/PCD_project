# BackportCheck

BackportCheck is a hybrid AI-assisted maintenance tool for the OpenStack ecosystem.  
It combines a Gerrit-integrated backport recommendation system with an executable benchmark for evaluating LLM-generated Infrastructure-as-Code backports.

The project has two main components:

1. **BackportCheck** — a Gerrit decision-support tool that predicts whether a submitted change should be recommended for backporting.
2. **IaC Backport Benchmark** — a Docker-based benchmark that evaluates whether local LLMs can generate correct stable-branch backports for OpenStack infrastructure and deployment-related changes.

---

## Key Results

### BackportCheck

BackportCheck uses an XGBoost classifier trained on OpenStack Gerrit data and enriched with an LLM explanation layer.

| Model | Accuracy | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 81.3% | 0.860 | 0.748 | 0.800 | 0.629 |

Average end-to-end response time:

| Mean latency | Max latency |
|---:|---:|
| 1.39s | 1.61s |

---

### IaC Backport Benchmark

The IaC benchmark evaluates local LLMs on two settings:

| Setting | Description |
|---|---|
| No-change control set | The merged stable version is equivalent to the merged master version |
| Adaptation-required set | The stable version differs from the master version and requires real adaptation |

Environment failures are excluded from the strict model comparison.

| Model | No-change set | Adaptation-required set |
|---|---:|---:|
| Qwen2.5-Coder-7B-Instruct-FP16 | 15/20 | 4/19 |
| DeepSeek-Coder-V2-Lite | 17/20 | 3/19 |

Qwen solved one adaptation-required instance that DeepSeek did not:

```text
openstack-neutron-stable-wallaby-I29a3910
