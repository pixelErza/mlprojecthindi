import os
import sys
from src.mlproject.exception import CustomException 
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

# Fixed variable assignments
host = os.getenv("host")
user = os.getenv("user") # Fixed from "user password"
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db  # Standard argument is 'database'
        )

        logging.info("Connection established successfully")
        
        df = pd.read_sql_query('select * from students', mydb)
        print(df.head())
        
        return df

    except Exception as ex:
        # Added 'sys' so your line numbers are reported correctly
        raise CustomException(ex, sys)