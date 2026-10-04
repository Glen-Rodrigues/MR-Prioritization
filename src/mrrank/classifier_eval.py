"""Phase 9b: honest evaluation of the meta-classifier.

Compares the original random 80/20 split against splits that hold out whole
MRs, whole mutants, or whole mutant families, and reports the majority-class
baseline, accuracy, F1, AUC, and ranking quality (Spearman vs true FDR).

Run:  python -m mrrank.classifier_eval
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split

from mrrank import config

NON_FEATURES = {"mr_id", "mutant_id", "label", "family"}


def make_clf() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                      random_state=config.SEED)


def mutant_family(mutant_id: str) -> str:
    return mutant_id.split("_layer")[0].rsplit("_run", 1)[0]


def classification_scores(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def out_of_fold(df, feats, splitter, groups) -> np.ndarray:
    p = np.zeros(len(df))
    for tr, te in splitter.split(df, df["label"], groups):
        clf = make_clf().fit(df.iloc[tr][feats], df.iloc[tr]["label"])
        p[te] = clf.predict_proba(df.iloc[te][feats])[:, 1]
    return p


def ranking_quality(df, p) -> dict:
    """How well do per-MR mean predicted kill probabilities rank the MRs?"""
    tmp = df.assign(p=p)
    pred = tmp.groupby("mr_id")["p"].mean()
    true = tmp.groupby("mr_id")["label"].mean().reindex(pred.index)
    rho = spearmanr(pred, true).statistic
    top2_pred = set(pred.sort_values(ascending=False).index[:2])
    top2_true = set(true.sort_values(ascending=False).index[:2])
    return {"spearman_vs_true_fdr": float(rho), "top2_overlap": len(top2_pred & top2_true)}


def main():
    df = pd.read_csv(config.OUTPUTS_DIR / "meta_classifier_features.csv")
    df["family"] = df["mutant_id"].map(mutant_family)

    feats = [c for c in df.columns if c not in NON_FEATURES and df[c].nunique() > 1]
    constant = [c for c in df.columns if c not in NON_FEATURES and df[c].nunique() <= 1]
    print(f"Rows: {len(df)}   Features used: {len(feats)}")
    if constant:
        print(f"Constant (uninformative) columns dropped: {constant}")

    y = df["label"]
    baseline = float(max(y.mean(), 1 - y.mean()))
    print(f"Kill rate: {y.mean():.3f}   Majority-class baseline accuracy: {baseline:.3f}\n")

    results = {"majority_baseline_accuracy": baseline, "kill_rate": float(y.mean()),
               "constant_columns": constant, "settings": {}}

    # 1) Original setting: random stratified 80/20 split
    X_tr, X_te, y_tr, y_te = train_test_split(df[feats], y, test_size=0.2,
                                              random_state=config.SEED, stratify=y)
    clf = make_clf().fit(X_tr, y_tr)
    results["settings"]["random_split (original)"] = classification_scores(y_te, clf.predict_proba(X_te)[:, 1])

    # 2-4) Grouped, out-of-fold settings
    grouped = {
        "leave-one-MR-out": (LeaveOneGroupOut(), df["mr_id"]),
        "held-out mutants (5-fold)": (GroupKFold(n_splits=5), df["mutant_id"]),
        "leave-one-mutant-family-out": (LeaveOneGroupOut(), df["family"]),
    }
    for name, (splitter, groups) in grouped.items():
        p = out_of_fold(df, feats, splitter, groups)
        results["settings"][name] = {**classification_scores(y, p), **ranking_quality(df, p)}

    print(f"{'Setting':<32}{'Acc':<8}{'F1':<8}{'AUC':<8}{'Spearman':<10}{'Top2'}")
    print("-" * 72)
    for name, r in results["settings"].items():
        sp = r.get("spearman_vs_true_fdr")
        t2 = r.get("top2_overlap")
        print(f"{name:<32}{r['accuracy']:<8.3f}{r['f1']:<8.3f}{r['auc']:<8.3f}"
              f"{(f'{sp:.3f}' if sp is not None else '-'):<10}{t2 if t2 is not None else '-'}")
    print(f"\nMajority baseline accuracy: {baseline:.3f}")

    out = config.OUTPUTS_DIR / "classifier_evaluation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
