## Transformation on data will be done here
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):  ## Creating pickle file for converting categorical feature to numerical..
        """This function is responsible for data transformation"""
        
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
        
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("Standard scaler",StandardScaler())
                ]

            )
            logging.info("numerical columns scaling completed")


            cat_pipeline = Pipeline(
                steps =  [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_hot _encoder",OneHotEncoder()),
                ]
            )

            logging.info("categorical columns encoding completed")

            logging.info("Standard scaling completed ")

            ## Combining numerical and categorical pipelines together

            preprocessor =  ColumnTransformer(
                [
                    ("Numerical pipeline",num_pipeline,numerical_columns),
                    ("Categorical Pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        
        except Exception as ex :
            raise CustomException(ex,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Test and train data")

            logging.info("Obtaining preprocessing objects")

            preprocessor_obj = self.get_data_transformer_obj() 

            target_column_name = "math_score"
            numerical_columns = ['reading_score', 'writing_score']
            input_features_train_df = train_df.drop(columns = target_column_name,axis = 1)
            target_feature_train_df = train_df[target_column_name]
            
            input_features_test_df = test_df.drop(columns = target_column_name,axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on test and train dataset")

            input_feature_train_transformed = preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_transformed = preprocessor_obj.transform(input_features_test_df)

            # It upgrades 1D arrays to 2D column vectors.
            # Then it concatenates them horizontally (along axis 1).
            train_transformed = np.c_[
                input_feature_train_transformed , np.array(target_feature_train_df)
            ]
            test_transformed = np.c_[input_feature_test_transformed,np.array(target_feature_test_df)]

            logging.info("Saving preprocessed objects")


            ## Saving preprocessor object as a pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (train_transformed,test_transformed , self.data_transformation_config.preprocessor_obj_file_path )



        except Exception as ex:
            raise CustomException(ex,sys)

