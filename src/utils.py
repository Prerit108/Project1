import os
import sys
print(sys.executable)

from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as ex:
        raise CustomException(ex,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        i = 0
        for model in list(models.values())   :
            model.fit(x_train,y_train)

            y_pred_test = model.predict(x_test)

            r2_score_test = r2_score(y_test,y_pred_test)

            report[list(models.keys())[i]] = r2_score_test
            i += 1

        return report
    

    except Exception as ex:
        raise CustomException(ex,sys)




    
            

