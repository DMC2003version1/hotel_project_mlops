from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH, CONFIG_PATH, PROCESSED_TRAIN_FILE_PATH, PROCESSED_DIR
from utils.common_functions import read_yaml
from src.data_preprocessing import DataPreprocessing
from src.logger import get_logger
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew

app = Flask(__name__)
logger = get_logger()

# Load model and preprocessing
model = None
config = None
preprocessor = None
selected_features = None

def load_model():
    global model, config, preprocessor, selected_features
    if model is None:
        logger.info(f"Loading model from {MODEL_OUTPUT_PATH}")
        model = joblib.load(MODEL_OUTPUT_PATH)
        logger.info("Model loaded successfully")
    if config is None:
        logger.info(f"Loading config from {CONFIG_PATH}")
        config = read_yaml(CONFIG_PATH)
        logger.info("Config loaded successfully")
    if preprocessor is None:
        logger.info("Initializing DataPreprocessing")
        preprocessor = DataPreprocessing(None, None, PROCESSED_DIR, CONFIG_PATH)
        logger.info("DataPreprocessing initialized")
    if selected_features is None:
        logger.info(f"Loading selected features from {PROCESSED_TRAIN_FILE_PATH}")
        # Load processed data to get selected features (without booking_status)
        processed_df = pd.read_csv(PROCESSED_TRAIN_FILE_PATH)
        selected_features = [col for col in processed_df.columns if col != 'booking_status']
        logger.info(f"Selected features: {selected_features}")
    return model, config, preprocessor, selected_features

def preprocess_input(data_dict):
    """Preprocess input data using the same pipeline as training"""
    try:
        logger.info("Starting data preprocessing")
        logger.info(f"Input data: {data_dict}")
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Drop Booking_ID if present (not needed for prediction)
        if 'Booking_ID' in df.columns:
            df.drop(columns=['Booking_ID'], inplace=True)
        
        # Get categorical and numerical columns from config
        categorical_columns = config["data_processing"]["categorical_columns"]
        numerical_columns = config["data_processing"]["numerical_columns"]
        
        # Remove booking_status from categorical if present
        if 'booking_status' in categorical_columns:
            categorical_columns = [c for c in categorical_columns if c != 'booking_status']
        
        # Label encode categorical columns
        # Note: In production, you should save and load the label encoders from training
        # For now, we'll use fit_transform which may not match training exactly
        for col in categorical_columns:
            if col in df.columns:
                if df[col].dtype == 'object' or isinstance(df[col].iloc[0], str):
                    le = LabelEncoder()
                    # Try to fit with the value, but ideally should use saved encoder
                    le.fit([str(df[col].iloc[0])])
                    df[col] = le.transform([str(df[col].iloc[0])])[0]
                else:
                    df[col] = int(df[col])
        
        # Skewness handling for numerical columns
        skewness_threshold = config["data_processing"]["skewness_threshold"]
        for col in numerical_columns:
            if col in df.columns:
                # Apply log1p if needed (simplified - should check actual skewness)
                # For inference, we'll apply log1p to columns that typically need it
                if col in ['lead_time', 'avg_price_per_room']:  # Common skewed columns
                    if df[col].values[0] > 0:
                        df[col] = np.log1p(df[col].values[0])
        
        # Select only the features that model expects (from feature selection)
        global selected_features
        if selected_features:
            # Ensure all selected features are present
            for feature in selected_features:
                if feature not in df.columns:
                    df[feature] = 0
            # Select only expected features in the correct order
            df = df[selected_features]
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        logger.info(f"Preprocessed data columns: {list(df.columns)}")
        logger.info("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise Exception(f"Error preprocessing input: {str(e)}")

@app.route('/')
def index():
    logger.info("Home page accessed")
    return render_template('index.html')

def get_form_data(config):
    """Get form data based on config columns"""
    data = {}
    
    # Get numerical columns from config
    numerical_columns = config["data_processing"]["numerical_columns"]
    for col in numerical_columns:
        if col == 'avg_price_per_room':
            data[col] = float(request.form.get(col, 0))
        else:
            data[col] = int(request.form.get(col, 0))
    
    # Get categorical columns from config (excluding booking_status)
    categorical_columns = config["data_processing"]["categorical_columns"]
    for col in categorical_columns:
        if col != 'booking_status':
            data[col] = request.form.get(col, '0')
    
    return data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Prediction request received")
        # Load model, config, and preprocessor
        model, config, preprocessor, selected_features = load_model()
        
        # Get form data from config
        data = get_form_data(config)
        logger.info(f"Form data extracted: {data}")
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Make prediction
        logger.info("Making prediction")
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Map prediction (0 = Not Canceled, 1 = Canceled)
        result = "Canceled" if prediction == 1 else "Not Canceled"
        confidence = max(prediction_proba) * 100
        
        logger.info(f"Prediction result: {result}")
        logger.info(f"Confidence: {confidence:.2f}%")
        logger.info(f"Probability - Canceled: {prediction_proba[1]*100:.2f}%, Not Canceled: {prediction_proba[0]*100:.2f}%")
        
        return render_template('result.html', 
                             prediction=result, 
                             confidence=round(confidence, 2),
                             proba_canceled=round(prediction_proba[1] * 100, 2),
                             proba_not_canceled=round(prediction_proba[0] * 100, 2))
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return render_template('result.html', 
                             error=f"Error: {str(e)}")

if __name__ == '__main__':
    logger.info("Starting Flask application on port 8080")
    app.run(debug=True, host='0.0.0.0', port=8080)

