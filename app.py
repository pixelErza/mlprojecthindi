from src.mlproject.logger import logging
from src.mlproject.exception import Custom_Exception
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation

import sys

if __name__ == "__main__":

    logging.info("The execution has started")

    try:
        # Data Ingestion
        data_ingestion = DataIngestion()

        train_data, test_data = data_ingestion.initiate_data_ingestion()

        logging.info("Data ingestion completed")

        # Data Transformation
        data_transformation = DataTransformation()

        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )

        logging.info("Data transformation completed")

        print("Pipeline completed successfully")

    except Exception as e:
        logging.info("An error occurred")
        raise Custom_Exception(e, sys)