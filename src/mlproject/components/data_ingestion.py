import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Custom imports from your project structure
from src.mlproject.exception import CustomException 
from src.mlproject.logger import logging
from src.mlproject.utils import read_sql_data

@dataclass
class DataIngestionConfig:
    # Corrected: os.path.join (singular) and removed double .path
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(os.path.join('notebook/data','raw.csv'))
            # 1. Reading the data from MySQL
            df = read_sql_data()
            logging.info("Reading completed from MySQL database")

            # 2. Create 'artifacts' folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 3. Save the raw data before splitting
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # 4. Split data into Train and Test sets
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 5. Save the split datasets (Corrected variable names and attribute spelling)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Passes the error and the system info to your custom exception handler
            raise CustomException(e, sys)