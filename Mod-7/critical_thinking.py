import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

CSV_URL = "https://raw.githubusercontent.com/venky14/Machine-Learning-with-Iris-Dataset/master/Iris.csv"


def load_iris_data():
    """Load local Iris.csv if present, otherwise load the assigned GitHub CSV URL.
    Fall back to sklearn Iris data only if neither CSV source is available.
    """
    try:
        df = pd.read_csv("Iris.csv")
        print("Loaded local Iris.csv copied from the assigned GitHub dataset.")
    except FileNotFoundError:
        try:
            df = pd.read_csv(CSV_URL)
            print("Loaded dataset from GitHub CSV URL.")
            X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
            y = df["Species"]
            return X, y
        except Exception as error:
            print("Could not load GitHub CSV. Using sklearn Iris dataset as a fallback.")
            print(f"Reason: {error}")
            iris = load_iris(as_frame=True)
            X = iris.data
            y = iris.target
            return X, y

    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    y = df["Species"]
    return X, y


def main():
    X, y = load_iris_data()

    # Shuffle folds to avoid class-order bias in the Iris dataset.
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    k_values = range(1, 31)
    mean_scores = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
        mean_scores.append(scores.mean())
        print(f"k={k:2d} | mean accuracy={scores.mean():.4f}")

    best_index = mean_scores.index(max(mean_scores))
    best_k = list(k_values)[best_index]
    best_score = mean_scores[best_index]

    print("\nBest Result")
    print(f"Optimal k value: {best_k}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), mean_scores, marker="o")
    plt.xlabel("k value")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.title("KNN Hyperparameter Tuning on Iris Dataset")
    plt.xticks(list(k_values))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("knn_k_tuning_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()