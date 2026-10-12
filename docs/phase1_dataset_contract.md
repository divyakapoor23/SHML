# Trial 1 Dataset Contract

## Purpose

This dataset contract defines the expected structure, characteristics,
and data-quality requirements of the healthy Trial 1 dataset.

The contract will serve as the reference state for detecting data failures
before the self-healing system applies corrective actions.

## 1. Prediction Setup

**Dataset Name:**   Default of Credit Card Clients dataset
**Dataset Source:**  UCI Machine Learning Repository 
**Number of Rows:**  30,000 observations 
**Number of Input Features:**  23
**Identifier Column:** ID
**Target Variable:**   Default payment next month

**Class 0:** Client does not default on payment in the next month.

**Class 1:** Client defaults on payment in the next month.

**Positive Class:** Class 1 (default) 

**Unit of Observation:** One credit-card client with their demographic, credit-limit, billing, and historical repayment information. 

**Intended User of the Prediction:**  
The credit-card issuer or financial institution, which could use the prediction to identify clients at higher risk of defaulting on their payment in the following month.

**Natural Time/Sequence Column:** No explicit timestamp column.

**Explanation:** The dataset contains ordered historical monthly variables for repayment status, bill amounts, and payment amounts, but it does not contain a separate date or timestamp column that orders the observations themselves.

### Prediction-Risk Assessment

**Could any feature directly reveal the target?**

No obvious predictor directly reveals the target. The target variable
`default payment next month` must be excluded from the input features.

Historical repayment variables such as `PAY_0` through `PAY_6` describe
previous repayment behavior and can be used for prediction, provided
they were available before the prediction of next-month default.


**Would every feature be available at prediction time?**

Yes. The predictor variables appear to contain demographic, credit-limit,
billing, and historical payment information that would already be available
to the financial institution before predicting default in the following month.

However, the timing and definitions of the variables should be verified
before modeling to ensure that no feature contains future information.


**Which class is the positive class?**

Class 1 — the client defaults on payment in the following month.

**Does recall matter especially for the positive class?**

Yes.

**Reasoning:** Recall is important because it measures how many of the clients
who actually default are correctly identified by the model. A false negative
occurs when the model predicts that a client will not default, but the client
actually defaults. Therefore, recall for Class 1 will be an important
evaluation metric for Trial 1.

**Does recall matter especially for the positive class?**

Yes. Class 1 represents default. Recall measures how many clients who actually
default are correctly identified. Missing an actual default produces a false
negative, so Class-1 recall is an important Trial 1 evaluation metric.




# Trial 1 Dataset Contract

## 1. Prediction Setup
# What we completed first

## 2. Data Dictionary
# What we just completed

## 3. Healthy Reference Profile
# What we're doing now

## 4. Initial Data-Quality Rules
# Next

## 5. Connection to Self-Healing
# Final section

## Trial 1 Decisions / Notes

## Completion Checklist


Rows: 30,000
Columns: 25
Input features: 23
Target:
    Class 0 = 77.88%
    Class 1 = 22.12%