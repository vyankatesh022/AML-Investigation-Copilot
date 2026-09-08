# FinGuard AI

FinGuard AI is an Anti-Money Laundering (AML) transaction monitoring assistant designed to assist compliance analysts with preliminary transaction investigations. It integrates deterministic compliance rules, machine learning anomaly detection, regulatory policy retrieval (RAG), and a multi-step investigation workflow to produce structured, evidentiary case reports.

---

## Project Overview

Financial institutions process millions of daily transactions, making manual transaction monitoring labor-intensive and prone to fatigue. **FinGuard AI** addresses this challenge by automating the preliminary stages of an AML compliance investigation:

1. **Transaction Ingestion & Validation:** Validates transactions and checks originator and beneficiary balance movements.
2. **Dual-Screening Risk Scoring & Explainability:** Combines deterministic compliance rules with a trained Scikit-Learn Random Forest model, providing structured rule conditions, observed values, and top ML feature importance signals.
3. **Evidence Aggregation & Traceability:** Collects chronological, verifiable evidence records across all stages (`EV-TX-...`, `EV-RULE-...`, `EV-ML-...`, `EV-CUST-...`, `EV-HIST-...`, `EV-RAG-...`) with source tracking and confidence scores.
4. **Regulatory Grounding (RAG):** Retrieves relevant clauses and guidance from FATF standards, FinCEN advisories, and internal bank policies with chunk ID traceability.
5. **AI Investigation Summary & Traceability Matrix:** Generates an evidence-grounded case summary with citations, an Evidence Traceability Matrix, separated AI analytical observations, and recommended disposition (`CLOSE_AS_FALSE_POSITIVE`, `REQUEST_INFORMATION_RFI`, or `ESCALATE_TO_SAR_COMMITTEE`).
6. **FastAPI Delivery:** Exposes data, explainable risk scoring, and end-to-end investigation execution through validated REST endpoints.

> **Note on Scope:** FinGuard AI is intended strictly for **preliminary investigation assistance**. It does not make autonomous compliance, legal, or SAR filing decisions.

---

## Features

The following features are fully implemented and verified in the codebase:

* **Transaction Data Ingestion & Validation:** Loads, deduplicates, and validates tabular transaction records using Pydantic schemas (`src/data/loader.py`).
* **Feature Extraction & Preprocessing:** Computes dynamic behavioral metrics including balance drain ratios, night transactions, and counterparty country risk (`src/data/preprocessor.py`).
* **Explainable Rule-Based Detection:** Evaluates transactions against 5 real-world AML compliance rules with conditions, observed values, configured thresholds, and risk contribution scores (`src/risk_engine/rules.py`).
* **Explainable ML Risk Classifier:** Random Forest classifier with 15-feature numeric vector tracking, top ranking signals with feature importances, anomaly probabilities, and statistical vs. crime distinction (`src/risk_engine/ml_model.py`).
* **Combined Risk Assessor:** Dual-screening synthesis weighting deterministic rule severities and ML anomaly scores into a composite rating with full explainability metadata (`src/risk_engine/assessor.py`).
* **Unified Evidence Traceability & Collector:** Thread-safe collector tracking chronological evidence items with unique IDs (`EV-TX`, `EV-RULE`, `EV-ML`, `EV-CUST`, `EV-HIST`, `EV-RAG`), source attribution, and confidence metrics (`src/evidence/`).
* **AML Policy Document Loader & Chunker:** Loads and segments FATF, FinCEN, and internal bank policies into searchable markdown chunks (`src/rag/document_loader.py`).
* **Vector Store & Traceable Semantic Retrieval:** TF-IDF and cosine-similarity vector store indexing policies with traceable chunk IDs, document titles, authorities, and similarity scores (`src/rag/vector_store.py`, `src/rag/retriever.py`).
* **Model Context Protocol (MCP) Tools:** Standardized tool interfaces for customer profile lookup, transaction history analysis, transaction lookup, and risk evaluation (`src/mcp_tools/`).
* **LangGraph Investigation Workflow:** Stateful 6-stage linear graph orchestrating transaction lookups, risk scoring, KYC enrichment, history summarization, policy retrieval, evidence accumulation, and LLM summary generation (`src/workflow/graph.py`).
* **Evidence-Grounded Synthesizer & Traceability Matrix:** Produces structured case narratives with a Section 9 Evidence Traceability Matrix, clearly distinguishing factual records from AI interpretations with advisory disclaimers (`src/workflow/llm.py`).
* **FastAPI REST API:** Fully typed REST endpoints with interactive Swagger UI documentation, pagination, query filtering, CORS support, and comprehensive evidence trace schemas (`src/api/app.py`).

---

## Project Architecture

```text
                     Transaction Dataset (CSV) & Customer Store (JSON)
                                            │
                                            ▼
                              [ Pydantic Data Layer ]
                              - TransactionDataLoader
                              - CustomerDataLoader
                              - TransactionPreprocessor
                              - SeedDataLoader (Synthetic Generator)
                                            │
                                            ▼
                              [ Dual-Screening Risk Engine ]
                              - RuleEngine (5 AML Rules + RuleExplanation)
                              - MLRiskClassifier (Random Forest + MLExplanation)
                              - TransactionRiskAssessor
                                            │
                                            ▼
                        [ Evidence Traceability Layer ]
                        - EvidenceCollector (EV-TX, EV-RULE, EV-ML, etc.)
                        - EvidenceItem & EvidenceType schemas
                        - Evidence Traceability Matrix Generator
                                            │
                                            ▼
                              [ LangGraph Investigation Workflow ]
                              Stage 1: Get Transaction Details (EV-TX)
                              Stage 2: Run Risk Assessment (EV-RULE, EV-ML)
                              Stage 3: Fetch Customer KYC Profile (EV-CUST)
                              Stage 4: Analyze Transaction History (EV-HIST)
                              Stage 5: Retrieve Regulatory Policies (EV-RAG)
                              Stage 6: Synthesize Investigation Summary (LLM)
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       [ MCP Tools Interface ]                       [ RAG Knowledge Base ]
       - get_customer_profile                        - FATF Recommendation 16
       - get_transaction_history                     - FinCEN SAR Advisory
       - get_transaction_details                     - Bank Internal AML Policy
       - get_transaction_risk_analysis               - Cosine Vector Store
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                              [ LLM Synthesis Engine ]
                              - Live API (Gemini / OpenAI)
                              - Grounded Offline Synthesizer
                              - Structured 8-Section Summary
                              - Section 9: Evidence Traceability Matrix
                              - Non-Autonomous Advisory Disclaimers
                                            │
                                            ▼
                              [ FastAPI REST API Layer ]
                              - /api/health
                              - /api/transactions
                              - /api/transactions/high-risk
                              - /api/transactions/{tx_id}
                              - /api/transactions/{tx_id}/risk
                              - /api/investigate
```

---

## Technology Stack

The project uses the following technologies verified from `requirements.txt` and application imports:

* **FastAPI (`fastapi>=0.115.0`):** REST API framework providing route handling, automatic OpenAPI/Swagger documentation, and dependency injection.
* **Uvicorn (`uvicorn>=0.30.0`):** ASGI web server executing the FastAPI application.
* **Pydantic & Pydantic-Settings (`pydantic>=2.8.0`, `pydantic-settings>=2.4.0`):** Data validation, request/response schema modeling, evidence domain modeling, and type-safe environment configuration.
* **Pandas (`pandas>=2.2.0`):** Tabular data processing, CSV ingestion, missing value filtering, and dataset manipulation.
* **NumPy (`numpy>=1.26.0`):** Numerical array operations and vector math for TF-IDF cosine similarity calculations.
* **Scikit-Learn (`scikit-learn>=1.5.0`):** `RandomForestClassifier` for ML risk scoring and `TfidfVectorizer` for policy indexing.
* **SciPy (`scipy>=1.13.0`):** Sparse matrix computations supporting TF-IDF vector operations and Scikit-Learn classifiers.
* **Joblib (`joblib>=1.4.0`):** Serialization and persistence of trained ML models (`models/risk_classifier.joblib`) and vector indices (`data/chroma_db/policy_vectors.joblib`).
* **HTTPX (`httpx>=0.27.0`):** HTTP client utilized by the FastAPI `TestClient` and for outbound LLM API requests.
* **Python-Dotenv (`python-dotenv>=1.0.1`):** Parsing `.env` files into environment variables.
* **Pytest & Pytest-Asyncio (`pytest>=8.0.0`, `pytest-asyncio>=0.23.0`):** Automated test runner and test assertion suite.

---

## Project Structure

```text
aml/
├── .env.example                      # Template for configuration and API keys
├── .gitignore                        # Excludes virtual environments, secrets, caches, and models
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependency specifications
│
├── data/
│   ├── policies/                     # AML regulatory policy markdown documents
│   │   ├── bank_internal_aml_policy.md
│   │   ├── fatf_recommendation.md
│   │   └── fincen_sar_advisory.md
│   ├── processed/                    # Customer KYC profile store
│   │   └── customers_sample.json
│   └── raw/                          # Synthetic transaction dataset (100 records)
│       └── transactions_sample.csv
│
├── models/
│   └── risk_classifier.joblib        # Trained Scikit-Learn Random Forest model artifact
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Centralized typed configuration and rule thresholds
│   ├── api/                          # FastAPI REST API layer
│   │   ├── __init__.py
│   │   ├── app.py                    # API routes, middleware, and route handlers
│   │   └── schemas.py                # Pydantic request and response schemas (with Evidence trace)
│   ├── data/                         # Data loading and preprocessing layer
│   │   ├── __init__.py
│   │   ├── loader.py                 # Transaction and customer data loaders
│   │   ├── preprocessor.py           # Feature engineering and ratio calculations
│   │   └── seed_data.py              # Synthetic transaction, KYC, and model training generator
│   ├── evidence/                     # Evidence traceability and audit trail module
│   │   ├── __init__.py
│   │   ├── collector.py              # Chronological EvidenceCollector and markdown table formatter
│   │   └── models.py                 # EvidenceItem and EvidenceType Pydantic domain models
│   ├── mcp_tools/                    # Model Context Protocol (MCP) tool definitions
│   │   ├── __init__.py
│   │   ├── schemas.py                # MCP tool input/output Pydantic schemas
│   │   ├── server.py                 # In-process MCP server dispatcher
│   │   └── tools.py                  # Concrete tool implementations
│   ├── rag/                          # Retrieval-Augmented Generation policy engine
│   │   ├── __init__.py
│   │   ├── document_loader.py        # Markdown policy loader and text chunker
│   │   ├── retriever.py              # Semantic query generator and citation formatter with Chunk IDs
│   │   └── vector_store.py           # Cosine TF-IDF vector store implementation
│   ├── risk_engine/                  # Hybrid AML risk evaluation engine
│   │   ├── __init__.py
│   │   ├── assessor.py               # Combined dual-screening risk assessor
│   │   ├── ml_model.py               # Random Forest risk classification model & MLExplanation
│   │   └── rules.py                  # Deterministic compliance rules & RuleExplanation
│   └── workflow/                     # LangGraph investigation orchestration
│       ├── __init__.py
│       ├── graph.py                  # Linear state graph coordinating investigation
│       ├── llm.py                    # LLM integration service with offline fallback
│       ├── nodes.py                  # Individual graph node functions with Evidence collection
│       ├── prompt.py                 # System and evidence prompt construction
│       └── state.py                  # Investigation state schemas
│
└── tests/                            # Automated test suite
    ├── __init__.py
    ├── conftest.py                   # Shared pytest fixtures
    ├── test_api.py                   # Integration tests for FastAPI endpoints
    ├── test_assessor.py              # Tests for combined dual-screening risk assessor
    ├── test_evaluation.py            # AI summary quality and RAG scenario evaluation
    ├── test_loader.py                # Tests for data loaders and feature extraction
    ├── test_mcp_tools.py             # Tests for MCP tool execution and schemas
    ├── test_ml_model.py              # Tests for ML model training, save/load, and inference
    ├── test_rag.py                   # Tests for document chunking and vector retrieval
    ├── test_rules.py                 # Tests for 5 compliance rules and boundary limits
    └── test_workflow.py              # Tests for LangGraph workflow execution
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vyankatesh022/AML-Investigation-Copilot.git
cd AML-Investigation-Copilot
```

### 2. Create and Activate a Python Virtual Environment
Requires Python 3.10+ (tested on Python 3.13):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` file to create your local `.env`:
```bash
cp .env.example .env
```

### 5. Datasets & Model Verification
The repository includes sample datasets and policies out of the box:
* Synthetic transaction records: `data/raw/transactions_sample.csv`
* Customer KYC profiles: `data/processed/customers_sample.json`
* Policy documents: `data/policies/*.md`
* Pre-trained ML model artifact: `models/risk_classifier.joblib`

To regenerate the synthetic datasets or retrain the baseline Random Forest model at any time, run:
```bash
python -m src.data.seed_data
```

---

## Environment Variables

All settings are managed via Pydantic Settings in `src/config.py`. Configure them in your `.env` file:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `API_HOST` | string | `0.0.0.0` | Host interface for FastAPI ASGI server |
| `API_PORT` | integer | `8000` | Port for FastAPI ASGI server |
| `ENVIRONMENT` | string | `development` | Operating environment (`development` / `production`) |
| `LOG_LEVEL` | string | `INFO` | Logging level |
| `GEMINI_API_KEY` | string | `""` (Optional) | API key for live Gemini LLM synthesis |
| `OPENAI_API_KEY` | string | `""` (Optional) | API key for live OpenAI LLM synthesis |
| `HIGH_VALUE_THRESHOLD` | float | `10000.0` | Monetary threshold triggering High Amount rule |
| `STRUCTURING_LOWER_BOUND` | float | `9000.0` | Lower boundary for Structuring detection |
| `STRUCTURING_UPPER_BOUND` | float | `9999.99` | Upper boundary for Structuring detection |
| `RAPID_DRAIN_RATIO` | float | `0.90` | Fraction of account balance drained triggering alert |
| `RAPID_VELOCITY_TX_LIMIT_24H` | integer | `3` | Maximum transactions per 24h before velocity flag |

> **Note:** Providing an LLM API key is optional. If neither key is provided, the application automatically uses its built-in grounded offline synthesizer.

---

## Running the Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

* **API Base URL:** `http://localhost:8000`
* **Interactive Swagger UI Documentation:** `http://localhost:8000/docs`
* **Alternative ReDoc Documentation:** `http://localhost:8000/redoc`

---

## API Endpoints

All endpoints return JSON responses.

### 1. Health Check
* **Method:** `GET`
* **Path:** `/api/health`
* **Purpose:** Operational status, version, and environment check.
* **Parameters:** None.
* **Response:**
  ```json
  {
    "status": "healthy",
    "version": "0.1.0",
    "environment": "development"
  }
  ```

### 2. List Transactions
* **Method:** `GET`
* **Path:** `/api/transactions`
* **Purpose:** Retrieve a list of transactions with optional risk filtering and limit.
* **Query Parameters:**
  * `limit` (int, default: 20, min: 1, max: 500): Number of records to return.
  * `risk_level` (string, optional): Filter by risk level (`LOW`, `MEDIUM`, or `HIGH`).
* **Response:** Returns `total_records`, `returned_records`, and array of transaction objects.

### 3. List High-Risk Transactions
* **Method:** `GET`
* **Path:** `/api/transactions/high-risk`
* **Purpose:** Returns transactions flagged as elevated or high risk, sorted by risk score descending.
* **Query Parameters:**
  * `limit` (int, default: 20, min: 1, max: 100): Maximum high-risk transactions to return.
* **Response:** Array of objects containing `transaction_id`, `customer_id`, `amount`, `transaction_type`, `counterparty_country`, `risk_level`, `risk_score`, and `triggered_rules`.

### 4. Get Single Transaction Details
* **Method:** `GET`
* **Path:** `/api/transactions/{transaction_id}`
* **Purpose:** Retrieve full transaction metadata and balance deltas by ID.
* **Path Parameters:** `transaction_id` (string, e.g. `TX-1001`).
* **Response:** `200 OK` with transaction object, or `404 Not Found` if the transaction does not exist.

### 5. Evaluate Transaction Risk with Explainability
* **Method:** `GET`
* **Path:** `/api/transactions/{transaction_id}/risk`
* **Purpose:** Run dual-screening (rules engine + ML model) for a transaction with complete explainability traces.
* **Path Parameters:** `transaction_id` (string, e.g. `TX-1003`).
* **Response:** Returns `final_risk_level`, `final_risk_score`, `is_flagged`, `rule_score`, `rule_severity`, `triggered_rules`, `ml_probability`, `ml_risk_level`, `top_ml_signals`, `explanation`, `rule_explanations`, and `ml_explanation`.
  ```json
  {
    "transaction_id": "TX-1003",
    "final_risk_level": "HIGH",
    "final_risk_score": 0.895,
    "is_flagged": true,
    "rule_score": 0.9,
    "rule_severity": "HIGH",
    "triggered_rules": ["Potential Currency Structuring (Smurfing)"],
    "ml_probability": 0.95,
    "ml_risk_level": "HIGH",
    "top_ml_signals": ["newbalanceDest (importance: 0.253)", "balance_drain_ratio (importance: 0.206)"],
    "explanation": "Transaction flagged for potential structuring corridor evasion.",
    "rule_explanations": [
      {
        "rule_id": "RULE_STRUCTURING",
        "rule_name": "Potential Currency Structuring (Smurfing)",
        "triggered": true,
        "severity": "HIGH",
        "risk_contribution_score": 0.9,
        "reason": "The transaction amount of $9,500.00 falls into the configured structuring corridor ($9,000.00 - $9,999.99).",
        "condition": "STRUCTURING_LOWER_BOUND <= amount <= STRUURING_UPPER_BOUND",
        "observed_values": {"amount": 9500.0},
        "configured_thresholds": {"lower_bound": 9000.0, "upper_bound": 9999.99}
      }
    ],
    "ml_explanation": {
      "model_name": "RandomForestClassifier (Tabular AML Risk Classifier)",
      "model_version": "1.0.0",
      "anomaly_probability": 0.95,
      "prediction_label": "Anomalous",
      "risk_level": "HIGH",
      "is_anomalous": true,
      "input_features": {"amount": 9500.0, "oldbalanceOrg": 12000.0},
      "top_signals": ["newbalanceDest (importance: 0.253)", "balance_drain_ratio (importance: 0.206)"],
      "risk_interpretation": "High statistical anomaly detected (95.0% probability). Feature patterns diverge from baseline.",
      "known_limitations": "Supervised Random Forest trained on synthetic tabular AML features. An anomaly score quantifies statistical deviation and does NOT confirm financial crime."
    }
  }
  ```

### 6. Run AI Investigation Workflow with Evidence Traceability
* **Method:** `POST`
* **Path:** `/api/investigate`
* **Purpose:** Trigger the complete multi-step investigation workflow for a transaction, capturing chronological evidence and audit trail.
* **Request Body:**
  ```json
  {
    "transaction_id": "TX-1003"
  }
  ```
* **Response:**
  ```json
  {
    "transaction_id": "TX-1003",
    "workflow_status": "COMPLETED",
    "final_risk_level": "HIGH",
    "final_risk_score": 0.895,
    "is_flagged": true,
    "triggered_rules": ["Potential Currency Structuring (Smurfing)"],
    "customer_id": "CUST-103",
    "investigation_summary": "# Transaction Investigation Summary: TX-1003\n\n## 1. Transaction Details\n[Evidence: EV-TX-TX-1003]...\n\n## 9. Evidence Traceability Matrix...",
    "recommended_action": "ESCALATE_TO_SAR_COMMITTEE",
    "error_message": null,
    "evidence_trace": [
      {
        "evidence_id": "EV-TX-TX-1003",
        "evidence_type": "TRANSACTION_DETAIL",
        "source": "TransactionsDatabase",
        "description": "Transaction TX-1003: TRANSFER of $9,500.00 via BRANCH to destination USA.",
        "related_transaction_id": "TX-1003",
        "related_customer_id": "CUST-103",
        "retrieval_timestamp": "2026-09-08T14:10:00",
        "confidence_or_score": 1.0,
        "confidence_type": "Database Record Match"
      },
      {
        "evidence_id": "EV-RULE-RULE_STRUCTURING",
        "evidence_type": "RULE_INDICATOR",
        "source": "DeterministicRuleEngine",
        "description": "Rule 'Potential Currency Structuring (Smurfing)' (HIGH severity)",
        "related_transaction_id": "TX-1003",
        "confidence_or_score": 0.9,
        "confidence_type": "Rule Severity Score"
      }
    ],
    "rule_explanations": [...],
    "ml_explanation": {...},
    "ai_interpretation": {
      "assessment_type": "AI-Assisted Preliminary Investigative Assessment",
      "evaluated_risk_level": "HIGH",
      "recommended_action": "ESCALATE_TO_SAR_COMMITTEE",
      "is_autonomous_decision": false,
      "disclaimer": "This assessment is preliminary AI-assisted guidance for human compliance officers. It does NOT constitute a final AML, legal, or compliance determination.",
      "evidence_count": 8,
      "referenced_evidence_ids": ["EV-TX-TX-1003", "EV-RULE-RULE_STRUCTURING", "EV-ML-TX-1003"]
    }
  }
  ```

---

## How the Investigation Workflow Works

When `/api/investigate` is called, the LangGraph state graph (`src/workflow/graph.py`) executes 6 sequential nodes with end-to-end evidence collection:

1. **Stage 1: Retrieve Transaction Details (`get_transaction_details`):**
   Calls MCP tool to fetch transaction metadata and balance deltas; records `EV-TX-{tx_id}` in `EvidenceCollector`. Halts safely if transaction does not exist.
2. **Stage 2: Evaluate Transaction Risk (`get_risk_analysis`):**
   Calls MCP tool to execute deterministic rules and ML classifier; records `EV-RULE-{rule_id}` and `EV-ML-{tx_id}` with feature importance and anomaly metrics.
3. **Stage 3: Retrieve Customer KYC Profile (`get_customer_profile`):**
   Calls MCP tool to look up customer profile, occupation, turnover, KYC tier, and PEP status; records `EV-CUST-{customer_id}`.
4. **Stage 4: Fetch Historical Account Patterns (`get_transaction_history`):**
   Calls MCP tool to inspect customer transaction history; records `EV-HIST-{customer_id}` with velocity, average amount, and counterparties.
5. **Stage 5: Retrieve Regulatory Policies (`retrieve_aml_policies`):**
   Queries the RAG vector store using identified risk flags; records `EV-RAG-{doc_id}-{chunk_id}` with document authority and TF-IDF similarity scores.
6. **Stage 6: Synthesize Investigation Summary (`generate_investigation_summary`):**
   Assembles all evidence into a structured case report containing:
   * 1. Transaction Details (cited with `EV-TX-...`)
   * 2. Customer Information (cited with `EV-CUST-...`)
   * 3. Transaction Risk Assessment (cited with `EV-RULE-...` and `EV-ML-...`)
   * 4. Historical Transaction Observations (cited with `EV-HIST-...`)
   * 5. Potential Risk Indicators
   * 6. Relevant AML Policy Information (cited with `EV-RAG-...`)
   * 7. AI-Generated Initial Assessment (clearly labeled AI analytical perspective)
   * 8. Recommended Next Investigation Steps (`CLOSE_AS_FALSE_POSITIVE`, `REQUEST_INFORMATION_RFI`, or `ESCALATE_TO_SAR_COMMITTEE`)
   * 9. Evidence Traceability Matrix (full Markdown table mapping every evidence ID, source, description, and confidence metric)

---

## Testing

The project includes an automated test suite with **91 tests** covering all layers:

```bash
.venv/bin/python -m pytest tests/ -v
```

### Test Coverage by File

| Test File | Description | Tests | Status |
| :--- | :--- | :---: | :---: |
| [`tests/test_loader.py`](tests/test_loader.py) | CSV loading, column validation, customer profile lookups, and feature extraction | 13 | PASSED |
| [`tests/test_rules.py`](tests/test_rules.py) | 5 AML heuristic rules, boundary corridor thresholds, and score aggregation | 14 | PASSED |
| [`tests/test_ml_model.py`](tests/test_ml_model.py) | Random Forest training, feature extraction, model persistence, and prediction | 6 | PASSED |
| [`tests/test_assessor.py`](tests/test_assessor.py) | Combined rules + ML dual-screening risk assessor logic | 5 | PASSED |
| [`tests/test_rag.py`](tests/test_rag.py) | Document chunking, vector indexing, similarity search, and citation formatting | 12 | PASSED |
| [`tests/test_mcp_tools.py`](tests/test_mcp_tools.py) | MCP tool execution, parameter validation, and server dispatch | 10 | PASSED |
| [`tests/test_workflow.py`](tests/test_workflow.py) | LangGraph state graph execution, evidence accumulation, and prompt generation | 9 | PASSED |
| [`tests/test_api.py`](tests/test_api.py) | FastAPI endpoints, filtering, pagination limits, 404 handling, and 422 validation | 13 | PASSED |
| [`tests/test_evaluation.py`](tests/test_evaluation.py) | AI summary quality scoring (6 criteria), RAG typology scenarios, and edge cases | 9 | PASSED |
| **Total** | **Full automated test suite** | **91** | **100% PASSED** |

---

## Security Considerations

The following security practices are implemented in the codebase:

* **Secret Isolation:** API keys and environment configurations are loaded via `.env` using Pydantic Settings. No secrets or credentials are hardcoded.
* **Git Exclusions:** `.gitignore` excludes `.env`, virtual environment directories (`.venv/`), Python cache files (`__pycache__/`), test caches (`.pytest_cache/`), model artifacts (`models/*.joblib`), and vector indices.
* **Input Validation:** All API inputs are validated using Pydantic schemas, rejecting invalid request bodies with structured `422 Unprocessable Entity` responses.
* **Safe Error Handling:** Nonexistent records return clear `404 Not Found` messages without leaking stack traces or internal server state.
* **Grounded RAG Context:** Policy text is demarcated as reference context, preventing prompt injection from overriding system guidelines.

---

## Limitations

* **Synthetic Sample Data:** Uses a synthetic dataset of 100 transactions and 20 customers designed for local testing rather than live banking transaction volumes.
* **Rule Coverage:** Implements 5 core AML typologies (Structuring, High Amount, Balance Drain, Sanctioned Jurisdiction, Velocity Anomaly). Enterprise compliance engines typically evaluate hundreds of rules.
* **Model Complexity:** Uses a tabular Random Forest classifier. It does not perform graph analysis (e.g., detecting multi-hop mule account rings).
* **Policy Knowledge Base:** The vector store indexes 3 sample regulatory documents (FATF Rec 16, FinCEN Advisory, Bank AML Policy) rather than an exhaustive global regulatory database.
* **Human Review Mandatory:** AI investigation summaries provide preliminary evidentiary drafts and must not replace human compliance officers.
* **No Built-in Authentication:** The FastAPI backend does not currently enforce user authentication or role-based access control (RBAC).

---

## Disclaimer

> FinGuard AI is an educational and portfolio project designed to support preliminary transaction investigations. The application's AI-generated analysis should not be considered a final AML, compliance, legal, or financial decision. Human review is required before taking any compliance-related action.

---

## Future Improvements

The following improvements are candidates for future development:

* **Graph-Based Network Analysis:** Implement graph analytics (e.g., NetworkX or Neo4j) to detect cyclical transfer patterns and multi-hop mule rings.
* **Expanded Rule Typologies:** Add rules for trade-based money laundering, round-trip transactions, and dormant account reactivations.
* **Authentication & Authorization:** Add JWT-based authentication and role-based access control for compliance analyst and admin roles.
* **Expanded Policy Store:** Index wider regulatory bodies including EU AMLD6, UK FCA regulations, and OFAC sanctions lists.
* **Web Frontend Interface:** Develop an interactive dashboard (e.g., Streamlit or React) for viewing alert queues and triggering investigations visually.
* **Containerization:** Provide Docker and docker-compose configurations for containerized deployment.

---

## Contribution

Contributions are welcome:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Ensure all tests pass (`.venv/bin/python -m pytest tests/ -v`).
4. Commit your changes (`git commit -m "add feature"`).
5. Push to the branch (`git push origin feature/my-feature`).
6. Open a Pull Request.
