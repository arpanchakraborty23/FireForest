import os,sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object,model_evaluate

@dataclass
class ModelTrainConfig:
    model_file_path=os.path.join('model','model.pkl')


class ModelTrain:
    def __init__(self):
        self.model_train_config=ModelTrainConfig()

    def initiating_model_train(self,train_array,test_array):
        try:
            logging.info('model train has started')
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])


            models={
                'Liner Regression':LinearRegression(),
                'Lasso Regression': Lasso(),
                'ElasticNet Regression': ElasticNet(),
                'Ridge Regeassor' : Ridge(),
                'RandomForest Regressor': RandomForestRegressor(),
                'DecisionTree Regressor' : DecisionTreeRegressor()
                }
            model_report:dict=model_evaluate(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)   
            print( model_report)

            print( model_report)

            
            logging.info(f'Model Report : {model_report}')

            print('\n====================================================================================\n')

            # 6. Find the best model
            best_model_score=max(sorted(model_report.values()))

            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Score : {best_model_score}')


            save_object(
                    file_path=self.model_train_config.model_file_path,
                    obj=best_model
                )

            return self.model_train_config.model_file_path


        except Exception as e:
            logging.info(f'Error in model train {str(e)}')
            raise CustomException(sys,e)    