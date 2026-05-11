import os
import sys
import pickle
import numpy as np
import pandas as pd
import pymysql
import urllib.parse

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

        quoted_password = urllib.parse.quote_plus(
            password
        )

        conn_str = (
            f"mysql+pymysql://{user}:"
            f"{quoted_password}@{host}/{db}"
        )

        engine = create_engine(conn_str)

        logging.info(
            "Connection established successfully"
        )

        df = pd.read_sql_query(
            "select * from students",
            engine
        )

        return df

    except Exception as ex:

        raise CustomException(ex, sys)


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:

            pickle.dump(obj, file_obj)

        logging.info(
            f"Object saved successfully at {file_path}"
        )

    except Exception as e:

        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):

    try:

        report = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]

            para = param[list(models.keys())[i]]

            gs = GridSearchCV(
                model,
                para,
                cv=3
            )

            gs.fit(X_train, y_train)

            model.set_params(
                **gs.best_params_
            )

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            report[
                list(models.keys())[i]
            ] = test_model_score

        return report

    except Exception as e:

        raise CustomException(e, sys)


def load_object(file_path):

    try:

        with open(file_path, "rb") as file_obj:

            return pickle.load(file_obj)

    except Exception as e:

        raise CustomException(e, sys)