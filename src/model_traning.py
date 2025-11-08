import os
import pandas as pd
import sys
import joblib 
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import *
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger()

class ModelTraining:
    def __init__(self, train_path, test_path, model_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = model_path
        self.params_dict = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_FOREST_PARAMS
        
    def load_and_split_data(self):
        try:
            logger.info("Loading and splitting data")
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_lightgbm_model(self, X_train, y_train):
        try:
            logger.info("Training LightGBM model")
            model = lgb.LGBMClassifier(
                random_state=self.random_search_params['random_state'],
                verbosity=-1  # Suppress warnings
            )
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.params_dict,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )
            logger.info("Starting random search for LightGBM model")
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            best_lightgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters: {best_params}")
            return best_lightgbm_model
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
        except Exception as e:
            raise CustomException(e, sys)
        
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logger.info("Saving model")
            joblib.dump(model, self.model_path)
            logger.info(f"Model saved at {self.model_path}")
        except Exception as e:
            raise CustomException(e, sys)
        
    def process(self):
        try:
            with mlflow.start_run():
                logger.info("Running model training")
                
                logger.info("Starting our MLFLOW experimentation")
                
                logger.info("Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(self.train_path, artifact_path="dataset")
                mlflow.log_artifact(self.test_path, artifact_path="dataset")
                
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lightgbm_model = self.train_lightgbm_model(X_train, y_train)
                
                evaluation_metrics = self.evaluate_model(best_lightgbm_model, X_test, y_test)
                logger.info(f"Evaluation metrics: {evaluation_metrics}")
                
                logger.info("Saving the model")
                self.save_model(best_lightgbm_model)
                
                logger.info("Logging the model to MLFLOW")
                mlflow.log_artifact(self.model_path, artifact_path="model")
                
                logger.info("Logging the model parameters to MLFLOW")
                mlflow.log_params(best_lightgbm_model.get_params())
                
                logger.info("Logging the evaluation metrics to MLFLOW")
                mlflow.log_metrics(evaluation_metrics)
                
                logger.info("Ending our MLFLOW experimentation")

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    model_training = ModelTraining(train_path=PROCESSED_TRAIN_FILE_PATH, test_path=PROCESSED_TEST_FILE_PATH, model_path=MODEL_OUTPUT_PATH)
    model_training.process()