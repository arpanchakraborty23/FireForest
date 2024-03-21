import os,sys
import pandas as pd 
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException
from src.utils.utils import import_data_from_mongo
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrain



@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifats','train.csv')
    test_data_path=os.path.join('artifats','test.csv')
    raw_data_path=os.path.join('artifats','raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('intiate data ingestion')

            #read data
            # df:pd.DataFrame=import_data_from_mongo(database_name='CSV',
            #                 collection_name='Clean_Algerian_forest_fires_dataset')

            df=pd.read_csv('experiment/data/Fire_Forest_Clean.csv')

            logging.info('data redad compelted')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path)
            

            logging.info(f'read data completed {df.head().to_string()}')

            train_data,test_data=train_test_split(df,test_size=0.29,random_state=0)

            train_data.to_csv(self.data_ingestion_config.train_data_path,header=True,index=False)

            test_data.to_csv(self.data_ingestion_config.test_data_path,header=True,index=False)

            logging.info('data ingestion completed')

            return (self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path
                )



        except Exception as e:
             
            raise CustomException(sys,e) from e  
            logging.info('Error occured: ',str(e) )

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()  

    transformation_obj=DataTransformation()
    train_arr,test_arr,_=transformation_obj.initiate_data_transformation(train_data,test_data)

    model_train=ModelTrain()
    print(model_train.initiating_model_train(train_arr,test_arr))