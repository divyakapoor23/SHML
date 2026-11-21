
import pandas as pd

from src.recovery import recover_missingness

X_sample = pd.DataFrame({
    "BILL_AMT1": [1000, None, 5000, None],
    "BILL_AMT2": [900, 2000, 4500, 3000]
})

diagnosis = {
    "feature": "BILL_AMT1",
    "severity": "moderate",
    "recovery_action": "Zero imputation"
}

X_recovered, recovery_log = recover_missingness(
    X_batch=X_sample,
    diagnosis=diagnosis,
    training_median=22225.5
)

print(X_recovered)
print(recovery_log)

assert X_recovered["BILL_AMT1"].isna().sum() == 0
assert X_recovered["BILL_AMT1"].tolist() == [
    1000, 0, 5000, 0
]

# Ensure the original batch wasn't modified
assert X_sample["BILL_AMT1"].isna().sum() == 2

print("Step 16A passed!")
