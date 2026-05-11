import os
import sys
from dataclasses import dataclass

import numpy as np
import dagshub
import mlflow
import mlflow.sklearn

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models


# Initialize DagsHub + MLflow
dagshub.init(
    repo_owner="pixelErza",
    repo_name="mlprojecthindi",
    mlflow=True
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):

        rmse = np.sqrt(
            mean_squared_error(actual, pred)
        )

        mae = mean_absolute_error(actual, pred)

        r2 = r2_score(actual, pred)

        return rmse, mae, r2

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing input data"
            )

            # Split dataset
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Models
            models = {

                "Random Forest": RandomForestRegressor(),

                "Decision Tree": DecisionTreeRegressor(),

                "Gradient Boosting": GradientBoostingRegressor(),

                "Linear Regression": LinearRegression(),

                "XGBRegressor": XGBRegressor(),

                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False
                ),

                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameters
            params = {

                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson"
                    ]
                },

                "Random Forest": {
                    "n_estimators": [
                        8, 16, 32, 64, 128, 256
                    ]
                },

                "Gradient Boosting": {
                    "learning_rate": [
                        0.1, 0.01, 0.05, 0.001
                    ],

                    "subsample": [
                        0.6, 0.7, 0.75,
                        0.8, 0.85, 0.9
                    ],

                    "n_estimators": [
                        8, 16, 32, 64, 128, 256
                    ]
                },

                "Linear Regression": {},

                "XGBRegressor": {
                    "learning_rate": [
                        0.1, 0.01, 0.05, 0.001
                    ],

                    "n_estimators": [
                        8, 16, 32, 64, 128, 256
                    ]
                },

                "CatBoost Regressor": {
                    "depth": [6, 8, 10],

                    "learning_rate": [
                        0.01, 0.05, 0.1
                    ],

                    "iterations": [
                        30, 50, 100
                    ]
                },

                "AdaBoost Regressor": {
                    "learning_rate": [
                        0.1, 0.01, 0.5, 0.001
                    ],

                    "n_estimators": [
                        8, 16, 32, 64, 128, 256
                    ]
                }
            }

            # Evaluate all models
            model_report = evaluate_models(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models,
                param=params
            )

            # Best model score
            best_model_score = max(
                model_report.values()
            )

            # Best model name
            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            # Best model object
            best_model = models[best_model_name]

            print(
                f"Best Model Found: {best_model_name}"
            )

            print(
                f"Best Model Score: {best_model_score}"
            )

            # Train best model
            best_model.fit(
                X_train,
                y_train
            )

            # Start MLflow run
            with mlflow.start_run():

                # Prediction
                predicted_qualities = best_model.predict(
                    X_test
                )

                # Evaluation metrics
                rmse, mae, r2 = self.eval_metrics(
                    y_test,
                    predicted_qualities
                )

                # Log parameter
                mlflow.log_param(
                    "best_model",
                    best_model_name
                )

                # Log metrics
                mlflow.log_metric(
                    "rmse",
                    rmse
                )

                mlflow.log_metric(
                    "mae",
                    mae
                )

                mlflow.log_metric(
                    "r2_score",
                    r2
                )

                # Log model
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model"
                )

            # Check model performance
            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found"
                )

            logging.info(
                "Best model found on training and testing dataset"
            )

            # Save model locally
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return r2

        except Exception as e:

            raise CustomException(e, sys)