# ============================================================
# PATH SETUP - MUST BE AT THE VERY TOP
# ============================================================
import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to get project root
project_root = os.path.dirname(os.path.dirname(current_dir))
# Add project root to Python path
sys.path.insert(0, project_root)

# Now these imports will work
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Prediction pipeline started")
            
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info(f"Loading model from: {model_path}")
            logging.info(f"Loading preprocessor from: {preprocessor_path}")
            
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            logging.info("Model and preprocessor loaded successfully")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            logging.info(f"Prediction completed: {preds[0]}")
            
            return preds
        
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            logging.info("Creating dataframe from input data")
            
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe created successfully")
            
            return df

        except Exception as e:
            logging.error(f"Error creating dataframe: {str(e)}")
            raise CustomException(e, sys)


# ============================================================
# RUN SEPARATELY
# ============================================================
if __name__ == "__main__":
    
    logging.info("=" * 50)
    logging.info("Starting Prediction Pipeline Test")
    logging.info("=" * 50)
    
    try:
        # Create sample data
        logging.info("Creating sample input data")
        custom_data = CustomData(
            gender="female",
            race_ethnicity="group B",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=72,
            writing_score=74
        )
        
        # Convert to dataframe
        logging.info("Converting to dataframe")
        data_df = custom_data.get_data_as_data_frame()
        
        print("\n" + "=" * 50)
        print("INPUT DATA:")
        print("=" * 50)
        print(data_df)
        print("\n")
        
        # Initialize prediction pipeline
        logging.info("Initializing prediction pipeline")
        predict_pipeline = PredictPipeline()
        
        # Make prediction
        logging.info("Making prediction")
        prediction = predict_pipeline.predict(data_df)
        
        # Show result
        print("=" * 50)
        print("PREDICTION RESULT:")
        print("=" * 50)
        print(f"Predicted Math Score: {prediction[0]:.2f}")
        print("=" * 50)
        
        logging.info(f"Final Prediction - Math Score: {prediction[0]:.2f}")
        logging.info("Prediction pipeline test completed successfully!")
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        print(f"Error: {str(e)}")