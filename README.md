# PolicyPulse baseline ML project

This project trains two exploratory models from `policypulse_raw.csv`:

- **Acceptance classifier:** predicts the `public_reaction` label (`Accepted` or `Rejected`).
- **Public-approval regressor:** estimates `public_approval_percent`.

The CSV has 147 rows, of which only 44 are labeled. This is enough to build and
exercise a baseline, but not enough to claim reliable predictions for policy
decisions. The approval estimate is a survey-opinion measure; the dataset has no
explicit real-world policy-impact outcome, so this project does **not** predict
causal or measured policy impact.

The acceptance classifier uses semi-supervised self-training: labeled training
rows seed a logistic-regression classifier, which adds high-confidence
pseudo-labels from `UNLABELLED` rows. The confidence threshold is 0.75. The
public-approval regressor remains supervised because approval values are absent
from unlabeled rows.

The models use policy descriptions and policy attributes. They deliberately
exclude acceptance labels, approval/disapproval percentages, survey details,
evidence text, and source URLs from the input features to reduce target leakage.
Evaluation uses a stratified 25% holdout of labeled rows; the holdout is not used
for pseudo-labeling or model fitting. The printed and saved `metrics.json` report
includes accuracy, balanced accuracy, macro/weighted F1, macro precision/recall,
confusion matrix, and approval MAE/RMSE. Results should be treated as
preliminary because the labeled dataset is small.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

```powershell
python train.py
```

To select another dataset or output folder:

```powershell
python train.py --data path\to\policies.csv --output-dir artifacts
```

The trained model files and holdout metrics are written to `artifacts\`.

## Predict

Pass a CSV containing new policy rows. The model uses the policy fields listed
in `policypulse_model.py`; missing fields are handled as unknown or missing.

```powershell
python predict.py path\to\new_policies.csv --output predictions.csv
```

The output includes the predicted reaction, estimated probability of acceptance,
and estimated public approval percentage. The approval estimate is not an
impact score.
