
# BackportCheck: A Hybrid AI Decision Support System for OpenStack Maintenance

**Version:** 1.0

## 1. Abstract

1. **BackportCheck** is a Machine Learning-based Decision Support System (DSS) designed to optimize the code maintenance workflow within the OpenStack ecosystem. It integrates directly into the Gerrit code review interface to predict the eligibility of changes for backporting (propagation to stable branches).

The system addresses the opaque nature of traditional maintenance tools by employing a **hybrid architecture**:
--  **Quantitative Risk Assessment:** A Gradient Boosting model (XGBoost) trained on rigorous software metrics and historical interaction data.
--  **Qualitative Explainability (XAI):** A Large Language Model (Llama 3) that interprets the risk vector to provide natural language justifications, bridging the gap between statistical probability and human reasoning.

2. **IaC Backport Benchmark** — a Docker-based benchmark that evaluates whether local LLMs can generate correct stable-branch backports for OpenStack infrastructure and deployment-related changes.


## 2. Repository Organization

The repository structure is organized to separate data ingestion, model training, backend inference, and frontend presentation.

```text
ROOT
├── backend_server/         # Production Inference Engine
│   ├── app.py              # Flask API entry point
│   ├── Dockerfile          # Containerization logic
│   ├── docker-compose.yml  # Orchestration service
│   ├── requirements.txt    # Python dependencies
│   ├── stats_complete.json # Historical Knowledge Base (Author/File interactions)
│   ├── threshold.txt       # Dynamic user-defined sensitivity threshold
│   └── Xgboost_optimized.json # Pre-trained predictive model
│
├── data/                   # Data Engineering Pipeline
│   ├── features_generator.py # Logic for Entropy, TextStat, and Regex extraction
│   ├── scraper.py          # Gerrit API crawler
│   ├── processed_data/     # Cleaned CSVs ready for training
│   └── raw_data/           # Raw JSONL extracts from OpenStack
│
├── extension/              # Client-Side Interface (Chrome/Chromium)
│   ├── content.js          # DOM manipulation & Backend communication
│   ├── manifest.json       # Manifest V3 configuration
│   ├── popup.html          # Settings UI structure
│   └── popup.js            # Settings logic (Threshold synchronization)
│
└── tools/                  # Research & Development Utilities
    ├── build_history.py    # Generates the initial stats_complete.json
    └── train_xgboost.py    # Training pipeline with TimeSeriesSplit validation
```

## 3. System Architecture & Methodology

The project implements a **Hybrid Decision Support System** that processes raw code changes through three distinct stages:

### 3.1. Data Processing & Historical Context
The system does not rely on static analysis alone. It reconstructs a stateful knowledge base from the OpenStack history to solve the "Cold Start" problem:

*   **Stateful Analysis (`stats_complete.json`):** A persistent engine tracks the long-term reliability of every author and the stability of every file path. This allows the system to generate **Trust Scores** and **Risk Probabilities** dynamically in real-time.
*   **Vectorization (`features_generator.py`):** Raw JSON metadata is converted into a mathematical vector using a robust set of software metrics, including **Shannon Entropy** (for complexity), **Churn Density**, and **Regex-based Intent Detection** (distinguishing bug fixes from refactoring).

### 3.2. Predictive Modeling (XGBoost)
The core risk assessment is performed by a **Gradient Boosting Classifier**, chosen for its ability to handle tabular data and non-linear feature interactions better than deep learning approaches in this domain.

*   **Temporal Validation:** We utilize **TimeSeriesSplit** rather than random K-Fold validation. This strictly respects the causality of software evolution (training on past data, testing on future data) to prevent data leakage.


### 3.3. Explainable AI (Llama 3)
To bridge the gap between numerical probability and human reasoning, the system employs a **Generative AI Agent** (Llama 3-70b) via the Groq API. Unlike simple template-based explanations, the system constructs a **structured decision context** for the LLM to analyze.

The prompt is dynamically assembled from three distinct data sources to ensure the explanation is grounded in reality:
1.  **Change Artifacts:** The raw **commit message** (for semantic intent) and the **list of modified files** (for architectural scope).
2.  **Predictive Outcome:** The XGBoost **confidence score**, the calculated **verdict** (Recommended/Rejected), and the active **user sensitivity threshold**.
3.  **Risk Vector:** The full 37-feature set

## 4. Installation & Usage

You can run the backend server using **Docker (Recommended)** or via **Manual Python Execution**.

### 4.1. Prerequisites
*   **API Key:** To enable the Generative AI explanation features, you must configure a Groq API key.
*   **Create Environment File:**
    Navigate to the `backend_server/` directory and create a `.env` file:
    ```bash
    GROQ_API_KEY=gsk_your_key_here
    ```

---

### 4.2. Option A: Running with Docker (Recommended)
This method ensures a consistent environment and automatically handles dependency management.

1.  **Build and Start:**
    From the root directory, run:
    ```bash
    docker-compose -f backend_server/docker-compose.yml up --build
    ```
2.  **Verification:**
    The API will initialize on `http://localhost:5000`. The container automatically validates the presence of the model and history files.

---

### 4.3. Option B: Manual Installation (Local Python)
Use this method for development or if Docker is not available.

1.  **Install Dependencies:**
    ```bash
    pip install -r backend_server/requirements.txt
    ```

2.  **Initialize Historical Database (Crucial):**
    Before running the app, you must generate the author trust database from the raw data.
    ```bash
    python tools/build_history.py
    ```
    *Ensure the resulting `stats_complete.json` is placed in the `backend_server/` directory.*

3.  **Start the Server:**
    ```bash
    python backend_server/app.py
    ```

---

### 4.4. Client-Side Extension Installation
1.  Open Google Chrome and navigate to `chrome://extensions/`.
2.  Enable **Developer Mode** (top right toggle).
3.  Click **Load Unpacked**.
4.  Select the `extension/` directory from this repository.

## 5. Usage Workflow

1.  Navigate to any Change page on **review.opendev.org** (OpenStack Gerrit).
2.  The **BackportCheck** panel will appear in the bottom-right corner.
3.  Click **"Analyze Eligibility"**.
4.  **Interpretation:**
    *   **Verdict Banner:** Green (Recommended) or Red (Not Recommended) based on your sensitivity threshold.
    *   **AI Insight:** A qualitative explanation of the risk factors.
    *   **Key Indicators:** A grid displaying Author Trust, File Fit, and Complexity.

## 6. Training Reproduction (For Reviewers)

To reproduce the scientific results or retrain the model on new data:

1.  **Data Extraction:**
    ```bash
    python data/scraper.py
    ```
2.  **Feature Engineering (Crucial):**
    Convert raw JSONL into the training dataset (`openstack_complete.csv`). 
    ```bash
    python data/features_generator.py
    ```

3.  **History Initialization (For App):**
    Generate the `stats_complete.json` lookup table used by the backend for real-time inference.
    ```bash
    python tools/build_history.py
    ```

4.  **Model Training:**
    Train the XGBoost classifier using Temporal Cross-Validation.
    ```bash
    python tools/train_xgboost.py
    ```
    *This script performs Hyperparameter Optimization and exports the `Xgboost_optimized.json` model to the backend directory.*
## 7. Technology Stack

*   **Inference Engine:** Python 3.10, Flask, XGBoost.
*   **Data Processing:** Pandas, NumPy, TextStat.
*   **Generative AI:** Llama 3 (via Groq Cloud).
*   **Frontend:** HTML5, CSS3, JavaScript (Manifest V3).
*   **Containerization:** Docker.
