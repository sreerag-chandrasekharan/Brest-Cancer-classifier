import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess(input_path: str, output_dir: str):
    # --- Load dataset
    data = pd.read_csv(input_path)

    # --- Split features and labels
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # --- Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Save processed data and scaler
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train_scaled).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test_scaled).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw cytosis dataset (CSV)")
    parser.add_argument("--outdir", type=str, default="data/processed_cytosis", help="Output dir for processed data")
    args = parser.parse_args()

    preprocess(args.input, args.outdir)
