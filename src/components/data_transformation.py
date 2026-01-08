## Main purpose of transformation is to do feature engineering, data cleaning, preprocessing.

from src.exception import CustomException
from src.logger import logging

import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path :str = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        '''
        This method return the object of data transformater.
        '''
        try:
            num_cols = ['reading_score'	,'writing_score']
            cat_cols = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            # numerical columns Pipeline
            num_pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                                ('scale',StandardScaler())
                                           ])
            logging.info('Numerical data type pipeline completed.')

            # categorical columns Pipeline
            cat_pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                            ('encode',OneHotEncoder(handle_unknown="ignore",
    sparse_output=False))
                                          ])
            
            logging.info('Categorical data type pipeline completed.')
            preprocessor = ColumnTransformer(transformers=[('numerical',  num_pipeline,num_cols),
                                                          ('categorical', cat_pipeline,cat_cols  )
                                                          ])
            logging.info("Column Transformer complete")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read the train and test data completed.")

            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_obj()

            target_col = 'math_score'

            input_feature_train_data = train_df.drop(target_col,axis=1)
            target_feature_train_data = train_df[target_col].values
            
            input_feature_test_data = test_df.drop(target_col,axis=1)
            target_feature_test_data = test_df[target_col].values

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            inp_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            inp_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)


            train_arr = np.c_[
                inp_feature_train_arr, np.array(target_feature_train_data)
            ]
            test_arr = np.c_[inp_feature_test_arr, np.array(target_feature_test_data)]

            
            logging.info(f"Saved preprocessing object.")

            save_obj(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )




            
        except Exception as e:
            raise CustomException(e)



