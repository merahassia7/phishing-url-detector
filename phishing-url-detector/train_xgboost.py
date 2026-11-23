"""
Clean XGBoost trainer for URL phishing detection.

Default behavior:
1) Load base dataset from datasets/prepared_urls.csv
2) Train XGBoost with app.py-compatible features
3) Save model artifacts to models/

Label convention in saved model:
  0 = safe, 1 = phishing
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import extract_features


def save_feature_importance(model: XGBClassifier, feature_names: list[str], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top_features = importance_df.head(15).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_features["feature"], top_features["importance"], color="#2f80ed")
        ax.set_title("XGBoost Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()

        png_path = os.path.join(save_dir, "xgb_feature_importance.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {png_path}")
    except Exception as exc:
        print(f"Feature importance image skipped: {exc}")


def save_confusion_matrix(cm, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    labels = ["safe", "phishing"]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("XGBoost Confusion Matrix")
        ax.set_xticks(range(len(labels)), labels=[f"predicted {label}" for label in labels])
        ax.set_yticks(range(len(labels)), labels=[f"actual {label}" for label in labels])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        fig.tight_layout()
        png_path = os.path.join(save_dir, "xgb_confusion_matrix.png")
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        print(f"Saved: {png_path}")
    except Exception as exc:
        print(f"Confusion matrix image skipped: {exc}")


def save_threshold_sweep(rows: list[dict[str, float]], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    threshold_df = pd.DataFrame(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="precision")
        ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="recall")
        ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="f1")
        ax.set_title("XGBoost Threshold Sweep")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        png_path = os.path.join(save_dir, "xgb_threshold_sweep.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {png_path}")
    except Exception as exc:
        print(f"Threshold sweep image skipped: {exc}")


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(base_dir, "datasets", "prepared_urls.csv"),
        help="Path to base CSV dataset with URL and label columns.",
    )
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--safe-label", type=int, default=0, help="Value used for safe URLs in dataset.")
    parser.add_argument("--phishing-label", type=int, default=1, help="Value used for phishing URLs in dataset.")
    parser.add_argument("--save-dir", default=os.path.join(base_dir, "models"), help="Directory to save model artifacts.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_labels(df: pd.DataFrame, safe_label: int, phishing_label: int) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Accept numeric and text labels from multiple data sources.
    label_map = {
        str(safe_label): 0,
        str(phishing_label): 1,
        "0": 0,
        "1": 1,
        "benign": 0,
        "safe": 0,
        "legit": 0,
        "legitimate": 0,
        "good": 0,
        "normal": 0,
        "malware": 1,
        "phishing": 1,
        "malicious": 1,
        "bad": 1,
        "fraud": 1,
        "scam": 1,
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    return df


def main() -> None:
    args = parse_args()

    print(f"Loading: {args.data}")
    df = pd.read_csv(args.data)[[args.url_col, args.label_col]].dropna()
    df.columns = ["url", "label"]
    df["url"] = df["url"].astype(str).str.strip()
    df = normalize_labels(df, args.safe_label, args.phishing_label)

    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    print(f"Rows after clean: {len(df):,}")
    print("Label counts:")
    print(df["label"].value_counts().sort_index().rename({0: "safe", 1: "phishing"}).to_string())

    print("Extracting features...")
    X = pd.DataFrame(df["url"].apply(extract_features).tolist())
    y = df["label"]
    feature_names = list(X.columns)
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = XGBClassifier(
        n_estimators=550,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )

    print("Training...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_table = pd.DataFrame(
        cm,
        index=["actual_safe", "actual_phishing"],
        columns=["predicted_safe", "predicted_phishing"],
    )
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion matrix [safe, phishing]:")
    print(cm_table.to_string())
    print(classification_report(y_test, y_pred, target_names=["safe", "phishing"]))

    print("Threshold sweep (phishing = class 1):")
    probs = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    threshold_rows = []
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        pred_th = (probs >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred_th, average="binary", pos_label=1
        )
        threshold_rows.append({"threshold": th, "precision": p, "recall": r, "f1": f1})
        print(f"th={th:.2f} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_confusion_matrix(cm, args.save_dir)
    save_feature_importance(model, feature_names, args.save_dir)
    save_threshold_sweep(threshold_rows, args.save_dir)
    model_path = os.path.join(args.save_dir, "phishing_model.pkl")
    features_path = os.path.join(args.save_dir, "xgb_feature_names.pkl")
    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)
    print(f"Saved: {model_path}")
    print(f"Saved: {features_path}")
    print(f"Model classes_: {list(model.classes_)}")


if __name__ == "__main__":
    main()
