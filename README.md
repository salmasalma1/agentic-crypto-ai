# CRYPTO AI — Agentic AI Sense–Reason–Act Final Project

CRYPTO AI analyzes suspicious encoded/cipher messages using the exact required academic architecture:

1. **Automated Data Ingestion** through Streamlit upload/default dataset and live text input.
2. **DBSCAN then FCM** for unsupervised clustering and pseudo-label support.
3. **SVM with RBF kernel** as the real-time cipher/state detector.
4. **Parallel Reasoning** through SHAP explainability and SARIMA forecasting.
5. **Action Layer** through Total Action Score and an autonomous Telegram alert, with every action logged.

The current project supports **Base64, Base32, and Morse**, plus **Hex and ASCII** as extra deterministic decoding families (see `utils/config.py`). The LLM role is optional and constrained: it only extracts `cypher_text` from messy text. It does not classify or decode.

## Folder Structure

```text
agentic_crypto_ai/
  app.py                  # Streamlit dashboard
  analyze_message.py      # CLI
  train_local_pipeline.py # retrains the models in artifacts/
  requirements.txt
  .env.example
  .streamlit/config.toml
  artifacts/              # saved scaler, SVM, DBSCAN, FCM and SHAP background
  data/
    raw/dataset.csv
    knowledge/cipher_knowledge.csv
    outputs/
  agent/
  pipeline/
  actions/
  utils/
```

## Setup on Windows PowerShell

Python 3.11 was used for the original environment.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The trained models are already in `artifacts/`. To retrain them from `data/raw/dataset.csv`, run `python train_local_pipeline.py`.

## Quick CLI Test

```powershell
python analyze_message.py "Please analyze this suspicious encoded message: JBSWY3DPEBLW64TMMQ====== and run it through CRYPTO AI." --sample-per-type 120 --execute-api-action --force-api-action
```

Expected result: the extractor finds `JBSWY3DPEBLW64TMMQ======`, SVM predicts a supported cipher type, the deterministic decoder attempts decoding, SHAP/SARIMA run, Total Action Score is calculated, and the API action is logged.

## Test Inputs

### Base32
```text
Please analyze this suspicious encoded message: JBSWY3DPEBLW64TMMQ====== and run it through CRYPTO AI.
```
Expected decoded text: `Hello World`.

### Base64
```text
Run CRYPTO AI on this encoded command: QW5hbHl6ZSB0aGUgY3J5cHRvIG1hcmtldCBhbmQgc2VuZCBhbiBhbGVydA==
```
Expected decoded text: `Analyze the crypto market and send an alert`.

### Morse
```text
Check this encoded signal: .... . .-.. .-.. --- / .-- --- .-. .-.. -..
```
Expected decoded text: `HELLO WORLD`.

## Environment Variables

Copy `.env.example` to `.env`. Do not share real keys.

Telegram action variables:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_API_URL=https://api.telegram.org
ACTION_LOG_PATH=data/outputs/action_log.csv
```

If the Telegram variables are missing, the project does **not** crash. The action is skipped and logged in `data/outputs/action_log.csv` with `success=False`. (The module is still named `actions/email_action.py` for backward compatibility; there is no email/SMTP sending.)

Optional LLM variables:

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-05-01-preview
```

Without these, regex extraction and fallback reporting are used.

Optional LangSmith tracing:

```env
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=
```

## Compliance Notes

- DBSCAN runs before FCM.
- SVM uses `kernel="rbf"`.
- SHAP runs through KernelExplainer when available and falls back to model sensitivity only if SHAP fails. The dashboard and CLI explicitly show `XAI Score = SVM Confidence × SHAP Importance`.
- SARIMA uses SARIMAX when available and a transparent naive fallback only if SARIMA fails.
- Dynamic reclustering calls the actual DBSCAN → FCM → label → SVM pipeline again when drift is detected and enabled.
- Total Action Score is calculated from Step 3 and Step 4 outputs: SVM confidence, XAI score, SARIMA forecast risk, drift score, and a small cipher-risk adjustment.
- API action is safe: secrets are not hardcoded and missing credentials do not crash the demo.

## Dataset

`data/raw/dataset.csv` comes from a public dataset. Each row holds a plain text and its Base64, Base32, Hex, Morse and ASCII encodings.

## Status

This was a solo final project by Salma Ashraf. It is not actively maintained, and I have not re-tested it end to end since publishing. The `agent/` folder has its own `requirements.txt` with pinned versions for the LangGraph orchestrator.


