import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "student_data.csv"

OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Step 1: Load dataset
df = pd.read_csv(DATA_PATH)

print("DATASET PREVIEW")
print(df.head())

print("\nDataset shape:", df.shape)


# Step 2: Prepare input features and target
features = [
    "Age",
    "Marks",
    "StudyHours",
    "Attendance"
]

X = df[features].apply(
    pd.to_numeric,
    errors="coerce"
)

y = df["Result"].astype(str).str.strip()


# Step 3: Check dataset
if X.isna().all().any():
    raise ValueError(
        "At least one feature column contains no numeric values."
    )

if y.nunique() < 2:
    raise ValueError(
        "The Result column must contain at least two classes."
    )

print("\nClass distribution:")
print(y.value_counts())


# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Step 5: Define three models
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),

    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )
    ),

    "KNN": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3)
    )
}


# Step 6: Train and evaluate models
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"\n{'=' * 40}")
    print(name)
    print("=" * 40)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save a confusion matrix for each model
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["Fail", "Pass"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"]
    )

    display.plot()

    plt.title(f"{name} - Confusion Matrix")

    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150
    )

    plt.close()


# Step 7: Create comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))


# Step 8: Save comparison results
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# Step 9: Create a bar chart
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

plot_df = results_df.set_index("Model")[metrics]

plot_df.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1)
)

plt.title("Comparison of Machine Learning Models")

plt.ylabel("Score")

plt.xlabel("Model")

plt.xticks(rotation=15)

plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=150
)

plt.close()


# Step 10: Save a text report
with open(
    OUTPUT_DIR / "comparison_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("MACHINE LEARNING MODEL COMPARISON\n")
    file.write("=" * 40 + "\n\n")

    file.write(
        "Target: Result (Pass/Fail)\n"
    )

    file.write(
        "Models: Logistic Regression, Decision Tree, KNN\n\n"
    )

    file.write(
        results_df.to_string(index=False)
    )

    file.write(
        "\n\nEvaluation uses a single held-out test set.\n"
    )

    file.write(
        "The sample dataset is small, so scores are illustrative.\n"
    )


print("\nPROJECT COMPLETED SUCCESSFULLY!")

print("\nGenerated files:")
print("1. model_comparison.csv")
print("2. model_comparison.png")
print("3. comparison_report.txt")
print("4. Individual model confusion matrices")