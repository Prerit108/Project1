## This module will handle reading datasets from databases or files
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass   ## Python auto-generates several special methods based on the class attributes.
## No need to manually define __init__ etc.
class Dataingestionconfig:
    train_data_path: str = os.path.join("artifacts","train.csv")   ## train_data_path is of str type
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","raw.csv")


class Dataingestion:
    def __init__(self):
        self.ingestion_config = Dataingestionconfig()   ## It will contain 3 values of the 3 paths

    def initiate_data_ingestion(self):   ## It will read the data from different databases 
        logging.info("entered the data ingestion method ")
        try:
            df = pd.read_csv("dataset/data/stud.csv")
            logging.info("Read the dataset and converted to dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            ## Making a folder at this path if it doesn't exist

            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)
            ## Making a path for saving the raw data 

            logging.info("Train test split started")
            train_set,test_set = train_test_split(df,test_size = 0.2,random_state = 23)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)

            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)

            logging.info("Ingestion of data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
    

if __name__ == "__main__":
    obj1 = Dataingestion()
    obj1.initiate_data_ingestion()






