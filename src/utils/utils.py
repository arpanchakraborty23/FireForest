import pandas as pd 
import os,sys
import numpy
from pymongo import MongoClient
from dotenv import load_dotenv
import pickle

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.logger import logging
from src.exception import CustomException


def import_data_from_mongo(database_name, collection_name):
    # Connect to MongoDB
    try:
        client =MongoClient(os.getenv('url'))
        # Select the database and collection
        logging.info('error in  db')
        db = client[database_name]
        logging.info('error in Mongo collection ')
        collection = db[collection_name]
        logging.info('error in Mongo find ')

        # Query to retrieve data from the collection
     
        x=collection.find()
        data=[]
        for i in x:
            print(i)
            data.append(i)
            

        df=pd.DataFrame(data)
        df.drop('_id',axis=1,inplace=True)
        return df

    except Exception as e :
        logging.info(f'error in Mongo db {str(e)}')
        raise CustomException(sys,e)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def model_evaluate(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        logging.info('model evaluation started')
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = r2_score(y_test, y_pred)*100

            logging.info(f'accuracy score for {list(models.keys())[i]}: {accuracy}')
           
            

            # Calculate mean absolute error
            mae = mean_absolute_error(y_test, y_pred)
       

            # Calculate mean squared error
            mse = mean_squared_error(y_test, y_pred)
          
            report[list(models.keys())[i]] =[
                f'Accuracy: {accuracy:.2f}%  MAE: {mae:.2f}%  MSE: {mse:.2f}%'   
            ]

        return report

    except Exception as e:
            logging.info(f'Error in utils {str(e)}')
            raise CustomException(sys,e)     