import os
import sys
print(sys.executable)

from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as ex:
        raise CustomException(ex,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}
        # Iterate through each model using its name and the model object
        for model_name, model in models.items():
            # Check if a parameter grid is defined for the current model in the params dict
            if model_name in params:
                para = params[model_name]
                
                randomcv = RandomizedSearchCV(estimator=model, param_distributions=para,
                                              n_jobs=-1, verbose=0, n_iter=10, cv=3)
                randomcv.fit(x_train, y_train)
                
                # Use the best estimator found by the search
                best_estimator = randomcv.best_estimator_
                y_pred_test = best_estimator.predict(x_test)

                test_model_score = r2_score(y_test, y_pred_test)
                report[model_name] = test_model_score

            else:
                # If no parameters are specified, just fit the model with its default settings
                model.fit(x_train, y_train)
                y_pred_test = model.predict(x_test)

                test_model_score = r2_score(y_test, y_pred_test)
                report[model_name] = test_model_score
        
        return report
    
    except Exception as ex:
        raise CustomException(ex,sys)


# This function is used to load the pickle files
def load_object(file_path):   
    try:
        with open(file_path,"rb") as file:
            return dill.load(file)
    except Exception as ex:
        raise CustomException(ex,sys)
        



    
            

