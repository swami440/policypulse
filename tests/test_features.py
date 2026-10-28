import unittest

import pandas as pd

from policypulse_model import FEATURE_COLUMNS, build_features


class BuildFeaturesTests(unittest.TestCase):
    def test_features_exclude_outcomes_and_retain_policy_inputs(self):
        source = pd.DataFrame(
            [
                {
                    "policy_name": "Example policy",
                    "policy_description": "A policy description",
                    "policy_objective": "An objective",
                    "policy_domain": "healthcare",
                    "policy_year": 2024,
                    "public_reaction": "Accepted",
                    "public_approval_percent": 99,
                }
            ]
        )

        features = build_features(source)

        self.assertEqual(len(features), 1)
        self.assertIn("policy_text", features.columns)
        self.assertIn("policy_domain", features.columns)
        self.assertNotIn("public_reaction", features.columns)
        self.assertNotIn("public_approval_percent", features.columns)
        self.assertTrue(set(FEATURE_COLUMNS).issuperset({"policy_name", "policy_year"}))
        self.assertEqual(features.iloc[0]["policy_text"], "Example policy A policy description An objective")

    def test_missing_policy_fields_are_supported(self):
        features = build_features(pd.DataFrame([{"policy_name": "Policy"}]))

        self.assertEqual(features.iloc[0]["policy_text"], "Policy")
        self.assertEqual(features.iloc[0]["policy_domain"], "unknown")
        self.assertTrue(pd.isna(features.iloc[0]["policy_year"]))


if __name__ == "__main__":
    unittest.main()
