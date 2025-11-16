import os
import pandas as pd
import yaml
from google.cloud import storage
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

# Load environment variables from .env file
load_dotenv()

logger = get_logger()

class DataIngestion:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.bucket_name = self.config["data_ingestion"]["bucket_name"]
        self.bucket_file_name = self.config["data_ingestion"]["bucket_file_name"]
        self.train_ratio = self.config["data_ingestion"]["train_ratio"]
        
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Raw directory created at: {RAW_DIR}")
        
    def download_csv_from_gcp(self):
        try:
            credentials_path = os.getenv("GCLOUD_KEY")
            client = storage.Client.from_service_account_json(credentials_path)
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            
            logger.info(f"Downloaded the csv file from GCP to {RAW_FILE_PATH}")
            
        except Exception as e:
            logger.error(f"Error downloading the csv file from GCP: {e}")
            raise CustomException(f"Failed to download the csv file from GCP: {str(e)}", e)
        
    def split_data(self):
        try:
            logger.info("Splitting the data into train and test sets")
            df = pd.read_csv(RAW_FILE_PATH)
            
            train_df, test_df = train_test_split(df, test_size=self.train_ratio, random_state=42)
            
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            
            logger.info("Data split successfully into train and test sets")
        except Exception as e:
            logger.error(f"Error splitting the data into train and test sets: {e}")
            raise CustomException(f"Failed to split the data into train and test sets: {str(e)}", e)

    def run_data_ingestion(self):
        try:
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(f"Failed to ingest data: {str(e)}", e)
            

if __name__ == "__main__":
    data_ingestion = DataIngestion(CONFIG_PATH)
    data_ingestion.run_data_ingestion()
