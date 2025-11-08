import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import *
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from scipy.stats import skew

logger = get_logger()

class DataPreprocessing:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config_path = config_path
        self.config = read_yaml(config_path)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        # Destination files for processed outputs
        self.processed_train_path = PROCESSED_TRAIN_FILE_PATH
        self.processed_test_path = PROCESSED_TEST_FILE_PATH


    def preprocess_data(self, df):
        try:
            logger.info("Preprocessing the data")
            
            logger.info("Dropping the unnecessary columns")
            
            df.drop(columns=['Booking_ID'] , inplace=True)
            df.drop_duplicates(inplace=True)
            
            categorical_columns = self.config["data_processing"]["categorical_columns"]
            numerical_columns = self.config["data_processing"]["numerical_columns"]
            
            logger.info("Label encoding the categorical columns")
            label_encoder = LabelEncoder()
            mapping = {}
            # Build and keep a mapping per categorical column from original labels to encoded integers.
            # This is useful for:
            # - auditability and debugging (know exactly how labels were encoded)
            # - reversing encodings for reporting/inference outputs if needed
            # - ensuring consistent encodings across train/inference when persisted
            
            for col in categorical_columns:
                df[col] = label_encoder.fit_transform(df[col])
                mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            
            logger.info("Label Mapping are: ")
            for col, mapping in mapping.items():
                logger.info(f"{col}: {mapping}")
            
            logger.info("Doing Skewness Handling")
            # Skewness handling: many models benefit when highly skewed numerical features
            # are closer to symmetric. We compute skewness and apply a log1p transform
            # (log(1+x)) to features exceeding a configurable threshold. log1p is
            # zero-safe and reduces the impact of large outliers while preserving order.
            skewness_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[numerical_columns].apply(lambda x: skew(x))
            
            for col in skewness[skewness > skewness_threshold].index:
                df[col] = np.log1p(df[col])
            
            return df

        except Exception as e:
            logger.error(f"Error preprocessing the data: {e}")
            raise CustomException("Failed to preprocess the data", e)
        
    def balance_data(self, df):
        try:
            logger.info("Balancing the data")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            df_res = pd.concat([X_res, y_res], axis=1)
            logger.info("Data balanced successfully")
            return df_res
        except Exception as e:
            logger.error(f"Error balancing the data: {e}")
            raise CustomException("Failed to balance the data", e)
        
    def select_features(self, df):
        try:
            logger.info("Selecting the features")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
            selector.fit(X, y)
            importances = selector.estimator_.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            })
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            num_of_features = self.config["data_processing"]["no_of_features"]
            top_features = feature_importance_df.head(num_of_features)["Feature"].tolist()
            logger.info(f"Top {num_of_features} features are: {top_features}")
            df_selected = df[top_features + ['booking_status']]
            logger.info(f"Selected {num_of_features} features successfully")
            return df_selected
        except Exception as e:
            logger.error(f"Error selecting the features: {e}")
            raise CustomException("Failed to select the features", e)
        
    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving the data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving the data to {file_path}: {e}")
            raise CustomException("Failed to save the data", e)
        
    def process(self):
        try:
            logger.info("Running the data processing")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]
            self.save_data(train_df, self.processed_train_path)
            self.save_data(test_df, self.processed_test_path)
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error running the data preprocessing: {e}")
            raise CustomException("Failed to run the data preprocessing", e)
            
if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    data_preprocessing.process()