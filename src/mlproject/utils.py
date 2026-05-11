import os
import sys
import pickle
import numpy as np
import pandas as pd
import pymysql
import urllib.parse  # Step 1: Added for password encoding
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

load_dotenv()

# Database credentials
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        # Step 2: Encode the password to handle special characters like '@'
        # This prevents the 'nodename nor servname provided' error
        quoted_password = urllib.parse.quote_plus(password)
        
        # Step 3: Create the SQLAlchemy engine with the safe password
        conn_str = f"mysql+pymysql://{user}:{quoted_password}@{host}/{db}"
        engine = create_engine(conn_str)

        logging.info("Connection established successfully using SQLAlchemy")
        
        df = pd.read_sql_query('select * from students', engine)
        return df

    except Exception as ex:
        raise CustomException(ex, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Function to perform hyperparameter tuning and model evaluation.
    Matches logic from Tutorial 9 at [11:36].
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Hyperparameter Tuning using GridSearchCV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Assigning best parameters to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Scoring
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)