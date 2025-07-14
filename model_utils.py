import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from feedback_utils import generate_feedback


# Custom label encoder wrapper for categorical columns
class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_maps = {}
        self.columns = None  # to keep track of column names
        self.fitted = False  # track if fit() has been called

    def fit(self, X, y=None):
        # X can be DataFrame or numpy array
        if hasattr(X, 'columns'):
            self.columns = X.columns
            for col in self.columns:
                unique_vals = X[col].astype(str).unique()
                self.label_maps[col] = {val: idx for idx, val in enumerate(unique_vals)}
        else:
            n_cols = X.shape[1]
            self.columns = [f"col_{i}" for i in range(n_cols)]
            for i, col in enumerate(self.columns):
                unique_vals = np.unique(X[:, i].astype(str))
                self.label_maps[col] = {val: idx for idx, val in enumerate(unique_vals)}
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("LabelEncoderWrapper instance is not fitted yet. Call 'fit' before 'transform'.")
        if hasattr(X, 'columns'):
            X_out = X.copy()
            for col in self.columns:
                mapping = self.label_maps.get(col, {})
                X_out[col] = X_out[col].astype(str).map(mapping).fillna(-1).astype(int)
            return X_out
        else:
            X_out = np.zeros_like(X, dtype=int)
            for i, col in enumerate(self.columns):
                mapping = self.label_maps.get(col, {})
                col_vals = X[:, i].astype(str)
                mapped_col = np.array([mapping.get(val, -1) for val in col_vals])
                X_out[:, i] = mapped_col
            return X_out

# Grade classification function (same as app.py expectations)
def classify_grade(score):
    if score >= 70:
        return "First Class"
    elif score >= 60:
        return "Second Class Upper"
    elif score >= 50:
        return "Second Class Lower"
    elif score >= 45:
        return "Third Class"
    elif score >= 40:
        return "Pass"
    else:
        return "Fail"


def preprocess_and_train(data_path="data.csv", model_path="grade_prediction_model.pkl"):
    df = pd.read_csv(data_path)

    # Drop rows without target exam score
    df.dropna(subset=["Exam_Score"], inplace=True)

    # Create target variable
    df["Grade_Class"] = df["Exam_Score"].apply(classify_grade)

    # Add engineered features exactly like in app.py
    df['Study_Attendance_Interaction'] = df['Hours_Studied'] * df['Attendance']
    df['Previous_Scores_Binned'] = pd.cut(
        df['Previous_Scores'],
        bins=[0, 60, 80, 100],
        labels=['Low', 'Medium', 'High']
    ).astype(str)  # convert category to string

    # Drop raw exam score since target created
    df.drop(columns=["Exam_Score"], inplace=True)

    # Features and target
    X = df.drop(columns=["Grade_Class"])
    y = df["Grade_Class"]

    print("Columns in X:", list(X.columns))  # Debug print for column names

    categorical_cols = [
        'School_Type', 'Teacher_Quality', 'Gender', 'Internet_Access', 'Learning_Disabilities',
        'Distance_from_Home', 'Previous_Scores_Binned', 'Parental_Involvement',
        'Parental_Education_Level', 'Family_Income', 'Access_to_Resources', 'Motivation_Level',
        'Peer_Influence', 'Extracurricular_Activities'
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Pipelines for numeric and categorical data
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('label_encoder', LabelEncoderWrapper())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ])

    # Full pipeline with classifier
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Split data for training/validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [5, 10, None],
    }
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print("Validation accuracy:", grid_search.score(X_valid, y_valid))

    # Save best modelstrea
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path="grade_prediction_model.pkl"):
    return joblib.load(model_path)

if __name__ == "__main__":
    preprocess_and_train("data.csv", "grade_prediction_model.pkl")
