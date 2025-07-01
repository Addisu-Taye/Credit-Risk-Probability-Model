# Date: 2025-06-26
# Created by: Addisu Taye
# Purpose: Train, evaluate, and track credit risk models.
# Key features:
# - Train/test split
# - GridSearchCV for hyperparameter tuning
# - MLflow tracking
# Task Number: 5

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os


class ModelTraining:
    def __init__(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Credit Risk Modeling")

    def prepare_data(self, file_path="data/processed/model_ready_data.csv"):
        df = pd.read_csv(file_path)
        X = df.drop(columns=['is_high_risk'])
        y = df['is_high_risk']
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def train_logistic_regression(self, X_train, y_train):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def train_random_forest(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

    def log_model_mlflow(self, model, model_name, params, metrics):
        with mlflow.start_run():
            mlflow.set_tag("model", model_name)
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            mlflow.sklearn.log_model(model, model_name)
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            mv = mlflow.register_model(model_uri, model_name)
        return mv.version

    def run_training_pipeline(self, model_type="random_forest"):
        X_train, X_test, y_train, y_test = self.prepare_data()

        if model_type == "logistic_regression":
            model, params, score = self.train_logistic_regression(X_train, y_train)
        elif model_type == "random_forest":
            model, params, score = self.train_random_forest(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        metrics = self.evaluate_model(model, X_test, y_test)
        version = self.log_model_mlflow(model, model_type, params, metrics)

        print(f"✅ Best {model_type} trained.")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        return model, metrics, version