# CRYPTO AI — Agentic AI Sense–Reason–Act Final Project

CRYPTO AI analyzes suspicious encoded/cipher messages using the exact required academic architecture:

1. **Automated Data Ingestion** through Streamlit upload/default dataset and live text input.
2. **DBSCAN then FCM** for unsupervised clustering and pseudo-label support.
3. **SVM with RBF kernel** as the real-time cipher/state detector.
4. **Parallel Reasoning** through SHAP explainability and SARIMA forecasting.
5. **Action Layer** through Total Action Score and autonomous email/API logging.

The current project supports **Base64, Base32, and Morse**. The LLM role is optional and constrained: it only extracts `cypher_text` from messy text. It does not classify or decode.

## Folder Structure

```text
CRYPTO_AI_FIXED/
  app.py
  analyze_message.py
  requirements.txt
  .env.example
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

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

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

Email action variables:

```env
EMAIL_USER=
EMAIL_PASSWORD=
RECEIVER_EMAIL=
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=465
```

If credentials are missing, the project does **not** crash. It logs a simulated autonomous API action in `data/outputs/action_log.csv`.

Optional LLM variables:

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-05-01-preview
```

Without these, regex extraction and fallback reporting are used.

## Compliance Notes

- DBSCAN runs before FCM.
- SVM uses `kernel="rbf"`.
- SHAP runs through KernelExplainer when available and falls back to model sensitivity only if SHAP fails. The dashboard and CLI explicitly show `XAI Score = SVM Confidence × SHAP Importance`.
- SARIMA uses SARIMAX when available and a transparent naive fallback only if SARIMA fails.
- Dynamic reclustering calls the actual DBSCAN → FCM → label → SVM pipeline again when drift is detected and enabled.
- Total Action Score is calculated from Step 3 and Step 4 outputs: SVM confidence, XAI score, SARIMA forecast risk, drift score, and a small cipher-risk adjustment.
- API action is safe: secrets are not hardcoded and missing credentials do not crash the demo.
