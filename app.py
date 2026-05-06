from src.mlproject.logger import logging
from src.mlproject.exception import Custom_Exception
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig

import sys
if __name__=="__main__":
    logging.info("the execution h has started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        
    except Exception as e: # <--- This defines 'e' as the error
        logging.info("An error occurred during data ingestion")
        # Now 'e' exists, so you can pass it to your Custom_Exception
        raise Custom_Exception(e, sys)