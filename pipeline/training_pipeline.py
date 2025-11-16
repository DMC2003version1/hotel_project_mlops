from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_traning import ModelTraining
from config.paths_config import *

if __name__ == "__main__":
    ## 1 Data ingestion:
    data_ingestion = DataIngestion(config_path=CONFIG_PATH)
    data_ingestion.run_data_ingestion()
    
    ## 2 Data processing:
    data_preprocessing = DataPreprocessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    data_preprocessing.process()
    
    ## 3 Model training:
    model_training = ModelTraining(train_path=PROCESSED_TRAIN_FILE_PATH, test_path=PROCESSED_TEST_FILE_PATH, model_path=MODEL_OUTPUT_PATH)
    model_training.process()
    