import unittest

import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier

from train import evaluate_models


class EvaluateModelsTests(unittest.TestCase):
    def test_self_training_uses_unlabeled_rows_and_reports_holdout_metrics(self):
        rows = []
        for label, word, approval in (
            ("Accepted", "popular beneficial", 75),
            ("Rejected", "unpopular costly", 25),
        ):
            for index in range(12):
                rows.append(
                    {
                        "policy_name": f"{word} policy {index}",
                        "policy_description": f"A {word} policy for communities",
                        "policy_year": 2020 + index,
                        "mandatory": index % 2,
                        "financial_burden": index + 1,
                        "benefit_level": index + 1,
                        "implementation_cost": index + 2,
                        "policy_intensity": index + 1,
                        "coverage_level": index + 1,
                        "essential_service": index % 2,
                        "label_status": "LABELLED",
                        "public_reaction": label,
                        "public_approval_percent": approval + index % 3,
                    }
                )
        for index in range(6):
            rows.append(
                {
                    "policy_name": f"popular beneficial policy {index}",
                    "policy_description": "A popular beneficial policy for communities",
                    "policy_year": 2024,
                    "mandatory": 1,
                    "financial_burden": 2,
                    "benefit_level": 5,
                    "implementation_cost": 3,
                    "policy_intensity": 4,
                    "coverage_level": 5,
                    "essential_service": 1,
                    "label_status": "UNLABELLED",
                }
            )

        metrics, acceptance_model, _ = evaluate_models(pd.DataFrame(rows))

        self.assertIsInstance(
            acceptance_model.named_steps["classifier"], SelfTrainingClassifier
        )
        self.assertEqual(metrics["semi_supervised"]["algorithm"], "SelfTrainingClassifier")
        self.assertEqual(metrics["semi_supervised"]["unlabeled_rows_available"], 6)
        self.assertIn("pseudo_labeled_rows_final_fit", metrics["semi_supervised"])
        self.assertEqual(metrics["holdout_rows"], 6)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "confusion_matrix",
        ):
            self.assertIn(metric, metrics["acceptance"])
        self.assertEqual(
            sum(sum(row) for row in metrics["acceptance"]["confusion_matrix"]), 6
        )


if __name__ == "__main__":
    unittest.main()
