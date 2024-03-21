import os,sys
import pandas as pd 
import numpy as np 
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_file_path=os.path.join('preprocess','preprocess.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def preprocess(self):
        try:
            preprocess_obj=Pipeline(
                steps=[('impute',SimpleImputer(strategy='median')),
                      ('scale',StandardScaler())
                      ]
            )
            return   preprocess_obj

        except Exception as e:
            logging.info(f'Error in prepeocess {str(e)}')     


    def initiate_data_transformation(self,train_data_path,test_data_path):
            try:
                logging.info('data transformation started')
                train_df=pd.read_csv(train_data_path)
                test_df=pd.read_csv(test_data_path)

                train_df.drop(['date','Unnamed: 0'],axis=1,inplace=True)
                test_df.drop(['date','Unnamed: 0'],axis=1,inplace=True)

                logging.info(f'data read completed {train_df.columns}')

            

                Target_col='FWI'
                

                # x_train
                input_feature_train_df=train_df.drop(columns=Target_col,axis=1)
                logging.info(f' {input_feature_train_df.columns}')
                # y_train
                target_feature_train_df=train_df[Target_col]

                # x_test
                input_feature_test_df=test_df.drop(columns=Target_col,axis=1)
                #y_test
                target_feature_test_df=test_df[Target_col]

                # print(input_feature_train_df.head())

               

                # scale data
                scale=self.preprocess()
                
                transform_input_features_train_df=scale.fit_transform(input_feature_train_df) #x_train
                transform_input_feature_test_df=scale.transform(input_feature_test_df ) # x_test

                train_arr=np.c_[transform_input_features_train_df,np.array(target_feature_train_df)]
                test_arr=np.c_[transform_input_feature_test_df,np.array(target_feature_test_df)]



                save_object(
                    file_path=self.data_transformation_config.preprocess_file_path,
                    obj=scale
                )
              
                logging.info("Exited initiate_data_transformation method of DataTransformation class")
                
                print(train_arr)

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocess_file_path
                )


            except Exception as e:
                logging.info(f'Error in data transormation {str(e)}')       