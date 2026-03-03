# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare machine learning models (Q6-Q8)

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()
X = dataset.X
y = dataset.y

# Binary target: North Central vs other
y = y.apply(lambda x: "North Central" if x in ["East North Central", "West North Central"] else "other")

# Train/test split (same as starter)
sample_idx = X.sample(n=1000, random_state=1).index
X_train = X.loc[sample_idx].reset_index(drop=True)
y_train = y.loc[sample_idx].reset_index(drop=True)
X_test = X.drop(sample_idx).reset_index(drop=True)
y_test = y.drop(sample_idx).reset_index(drop=True)

# %% [markdown]
# ## Load models

# %%
import joblib
from midwest_survey_models.transformers import NumericalStabilizer  # needed for unpickling pipelines

model_lr = joblib.load("../model_logistic_regression.pkl")
model_rf = joblib.load("../model_random_forest.pkl")
model_gb = joblib.load("../model_gradient_boosting.pkl")

models = {
    "Logistic Regression": model_lr,
    "Random Forest": model_rf,
    "Gradient Boosting": model_gb,
}

POS = "North Central"

# %% [markdown]
# ## Q6

# %%
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def eval_model_on_test(name, model, X_test, y_test, pos_label=POS):
    y_pred = model.predict(X_test)

    # get precision/recall/f1 for pos_label
    labels = [pos_label, "other"]
    # precision_recall_fscore_support with labels order
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average=None, zero_division=0
    )
    # index 0 corresponds to pos_label
    precision_pos, recall_pos, f1_pos = p[0], r[0], f1[0]

    cm = confusion_matrix(y_test, y_pred, labels=[pos_label, "other"])
    # cm = [[TP, FN],
    #       [FP, TN]] with this labels order? Actually:
    # rows = true, cols = pred
    # labels=[pos, other]
    # => cm[0,0]=TP, cm[0,1]=FN, cm[1,0]=FP, cm[1,1]=TN
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    return {
        "name": name,
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "f1_pos": float(f1_pos),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }

results = [eval_model_on_test(n, m, X_test, y_test) for n, m in models.items()]

print("=== Test-set metrics (positive class = 'North Central') ===")
for r in results:
    print(f"\n[{r['name']}]")
    print(f"Precision: {r['precision_pos']:.4f}")
    print(f"Recall:    {r['recall_pos']:.4f}")
    print(f"F1:        {r['f1_pos']:.4f}")
    print(f"Confusion (TP,FN,FP,TN): {r['tp']}, {r['fn']}, {r['fp']}, {r['tn']}")

# Answer Q6
best_recall = max(results, key=lambda d: d["recall_pos"])
print("\n=== Q6 Answer ===")
print("Best recall model:", best_recall["name"], "with recall =", f"{best_recall['recall_pos']:.4f}")

# %% [markdown]
# ## Q7
# Costs / gains:
# - FP = -10
# - FN = -1
# - TP = +5
# - TN = +2

# %%
def practical_score(tp, fn, fp, tn):
    return 5 * tp + 2 * tn - 10 * fp - 1 * fn

for r in results:
    r["practical_score"] = practical_score(r["tp"], r["fn"], r["fp"], r["tn"])

best_practical = max(results, key=lambda d: d["practical_score"])
print("\n=== Q7 Answer ===")
for r in results:
    print(f"{r['name']}: practical_score = {r['practical_score']}")
print("Best practical application model:", best_practical["name"])

# %% [markdown]
# ## Q8
# We'll use cross_validate to get both train_score and test_score (accuracy).

# %%
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

cv_results = {}
for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=None,
    )
    train_mean = float(np.mean(scores["train_score"]))
    test_mean = float(np.mean(scores["test_score"]))
    gap = train_mean - test_mean
    cv_results[name] = {"train_mean": train_mean, "test_mean": test_mean, "gap": gap}

print("\n=== CV accuracy (train vs test) ===")
for name, d in cv_results.items():
    print(f"{name}: train={d['train_mean']:.4f}, test={d['test_mean']:.4f}, gap={d['gap']:.4f}")

best_generalizes = min(cv_results.items(), key=lambda kv: abs(kv[1]["gap"]))  # smallest gap
most_overfit = max(cv_results.items(), key=lambda kv: kv[1]["gap"])          # largest gap (train >> test)

print("\n=== Q8 Answer ===")
print("Best generalization (smallest train-test gap):", best_generalizes[0], "gap =", f"{best_generalizes[1]['gap']:.4f}")
print("Most likely overfitting (largest gap):", most_overfit[0], "gap =", f"{most_overfit[1]['gap']:.4f}")

# %% [markdown]
# ## Final suggestion (optional)
# Choose a model for real application: often the best practical_score.
