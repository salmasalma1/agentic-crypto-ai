#!/usr/bin/env python3
"""CLI smoke test for the CRYPTO AI pipeline."""
import argparse
from pipeline.pipeline_runner import run_crypto_ai_pipeline


def main():
    parser = argparse.ArgumentParser(description="Analyze one encoded message through CRYPTO AI.")
    parser.add_argument("message", nargs="?", default="Please analyze this suspicious encoded message: JBSWY3DPEBLW64TMMQ====== and run it through CRYPTO AI.")
    parser.add_argument("--sample-per-type", type=int, default=120, help="Rows per cipher type for quick CLI tests.")
    parser.add_argument("--execute-api-action", action="store_true", help="Allow API action/logging.")
    parser.add_argument("--force-api-action", action="store_true", help="Force API action for demo.")
    args = parser.parse_args()

    result = run_crypto_ai_pipeline(
        messy_text=args.message,
        execute_api_action=args.execute_api_action,
        force_api_action=args.force_api_action,
        enable_dynamic_reclustering=True,
        max_rows_per_type=args.sample_per_type,
    )

    print("=" * 80)
    print("CRYPTO AI PIPELINE RESULT")
    print("=" * 80)
    for i, item in enumerate(result["timeline"], start=1):
        print(f"{i:02d}. [{item['status'].upper()}] {item['step']} - {item.get('details', '')}")
    print("-" * 80)
    print("Extracted cypher text:", result["extraction_result"].get("cypher_text", ""))
    print("SVM predicted cipher:", result["extracted_prediction"])
    print("SVM confidence:", round(float(result["extracted_confidence"]), 4))
    print("Decoded plain text:", result.get("decoded_plain_text") or "Not decoded")
    print("SHAP importance score:", result["shap_result"].get("shap_score"))
    print("XAI score = confidence x importance:", result.get("xai_score"))
    print("Forecast risk score:", result["sarima_result"].get("forecast_risk_score"))
    print("Drift:", result["drift_result"])
    print("Total action score:", result["total_action_score"])
    print("Action required:", result["action_required"])
    print("API result:", result["api_result"].get("message"))
    print("=" * 80)


if __name__ == "__main__":
    main()
