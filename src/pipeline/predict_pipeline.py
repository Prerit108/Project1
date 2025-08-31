import sys
import os
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

## Prediction of the new data dictionary by importing the model and preprocessor
    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predicted_data = model.predict(data_scaled)

            logging.info("Prediction done for the new enterd data")
            return predicted_data
        except Exception as ex:
            raise CustomException(ex,sys) 



## Creating a class to handle the new data entered by the user in the html and convert it into a dict
class CustomData:
    def __init__(self,gender : str
                 , race_ethnicity : str
                 , parental_level_of_education : str, 
                 lunch : str, test_preparation_course : str,
                 reading_score : int,writing_score : int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            "gender":[self.gender] ,
            "race_ethnicity":[self.race_ethnicity] ,
            "parental_level_of_education" : [self.parental_level_of_education] ,
            "lunch" :[self.lunch],
            "test_preparation_course" :[self.test_preparation_course] ,
            "reading_score": [self.reading_score] ,
            "writing_score" : [self.writing_score] 
            }

            logging.info("Dataframe for new entered data created")

            return pd.DataFrame(custom_data_input_dict)

        except Exception as ex:
            raise CustomException(ex,sys) 
