import os
import sys

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object,evaluate_model


from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_data,test_data):
        try:
            logging.info("Splitting train test")

            x_train,x_test,y_train,y_test = ( 
                train_data[:,:-1],
                test_data[:,:-1],
                train_data[:,-1],
                test_data[:,-1]
            )

            models = {

                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
            }

            model_report:dict = evaluate_model(x_train = x_train,y_train = y_train,x_test = x_test,y_test=y_test,models = models) 

            ## To get the best model
            best_model_score = max(sorted(model_report.values()))

            ## To get the name of the best model
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("No Best model found",sys)
            
            logging.info("Model training and selection completed")
            
            save_object(
                file_path = ModelTrainerConfig.trained_model_file_path,
                obj = best_model
            )
            

            predicted_output = best_model.predict(x_test)

            score_r2 = r2_score(y_test,predicted_output)

            return score_r2

        except Exception as ex:
            raise CustomException(ex,sys) 
 


