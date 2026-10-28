"""Generate acceptance and public-approval estimates for policy rows."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from policypulse_model import build_features, load_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV containing policy rows to score.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).with_name("artifacts"),
        help="Directory containing models created by train.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Path where prediction results will be written.",
    )
    args = parser.parse_args()

    acceptance_model = joblib.load(args.model_dir / "acceptance_model.joblib")
    approval_model = joblib.load(args.model_dir / "approval_model.joblib")
    policies = load_csv(args.input_csv)
    features = build_features(policies)

    results = pd.DataFrame()
    for column in ("policy_id", "policy_name"):
        if column in policies.columns:
            results[column] = policies[column]
    results["predicted_public_reaction"] = acceptance_model.predict(features)
    accepted_index = list(acceptance_model.classes_).index("Accepted")
    results["predicted_acceptance_probability"] = acceptance_model.predict_proba(
        features
    )[:, accepted_index]
    results["predicted_public_approval_percent"] = (
        approval_model.predict(features).clip(0, 100)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
