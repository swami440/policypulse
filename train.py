"""Train acceptance and public-approval baseline models for PolicyPulse."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from policypulse_model import (
    ACCEPTANCE_LABELS,
    build_features,
    load_csv,
    make_preprocessor,
)


RANDOM_STATE = 42


def evaluate_models(data: pd.DataFrame) -> tuple[dict, Pipeline, Pipeline]:
    """Evaluate on a stratified holdout, then fit final models on all labeled rows."""
    labelled = data.loc[
        data["label_status"].eq("LABELLED")
        & data["public_reaction"].isin(ACCEPTANCE_LABELS)
    ].copy()
    if labelled.empty:
        raise ValueError("No rows have a usable LABELLED public_reaction value.")
    if labelled["public_reaction"].nunique() < 2:
        raise ValueError("At least two acceptance classes are required to train.")

    approval = pd.to_numeric(labelled["public_approval_percent"], errors="coerce")
    approval_mask = approval.notna()
    if approval_mask.sum() < 4:
        raise ValueError("At least four labeled public_approval_percent values are required.")

    features = build_features(labelled)
    labels = labelled["public_reaction"].astype(str)
    class_counts = labels.value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each acceptance class needs at least two labeled examples.")

    train_idx, test_idx = train_test_split(
        labelled.index,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_x, test_x = features.loc[train_idx], features.loc[test_idx]
    train_y, test_y = labels.loc[train_idx], labels.loc[test_idx]

    acceptance_model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    C=0.5,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    acceptance_model.fit(train_x, train_y)
    predicted_labels = acceptance_model.predict(test_x)

    approval_train_mask = approval.loc[train_idx].notna()
    approval_test_mask = approval.loc[test_idx].notna()
    if approval_test_mask.sum() < 2:
        raise ValueError("The holdout needs at least two approval values for evaluation.")
    approval_model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )
    approval_model.fit(
        features.loc[train_idx[approval_train_mask.to_numpy()]],
        approval.loc[train_idx[approval_train_mask.to_numpy()]],
    )
    approval_test_x = features.loc[test_idx[approval_test_mask.to_numpy()]]
    approval_test_y = approval.loc[test_idx[approval_test_mask.to_numpy()]]
    predicted_approval = approval_model.predict(approval_test_x).clip(0, 100)

    metrics = {
        "dataset_rows": int(len(data)),
        "labelled_rows": int(len(labelled)),
        "acceptance_class_counts": {
            str(label): int(count) for label, count in class_counts.items()
        },
        "approval_rows": int(approval_mask.sum()),
        "holdout_rows": int(len(test_idx)),
        "acceptance": {
            "accuracy": float(accuracy_score(test_y, predicted_labels)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, predicted_labels)),
            "macro_f1": float(f1_score(test_y, predicted_labels, average="macro")),
        },
        "public_approval_percent": {
            "mae_percentage_points": float(
                mean_absolute_error(approval_test_y, predicted_approval)
            ),
            "rmse_percentage_points": float(
                mean_squared_error(approval_test_y, predicted_approval) ** 0.5
            ),
            "evaluation_rows": int(len(approval_test_y)),
        },
        "warning": (
            "Exploratory metrics only: the labeled dataset is small and may not "
            "represent all policies or survey populations."
        ),
    }

    acceptance_model.fit(features, labels)
    approval_model.fit(features.loc[approval_mask], approval.loc[approval_mask])
    return metrics, acceptance_model, approval_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("policypulse_raw.csv"),
        help="Path to the source policy CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("artifacts"),
        help="Directory for trained models and evaluation metrics.",
    )
    args = parser.parse_args()

    data = load_csv(args.data)
    metrics, acceptance_model, approval_model = evaluate_models(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(acceptance_model, args.output_dir / "acceptance_model.joblib")
    joblib.dump(approval_model, args.output_dir / "approval_model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nModels and metrics saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
