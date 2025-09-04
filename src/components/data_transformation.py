from dataclasses import dataclass
import sys
import os

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.loggers import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is reponsible for the data transformation
        
        '''
        try: 
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Columns:{numerical_columns}")
            logging.info(f"Categorical Columns:{categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("Num",num_pipeline,numerical_columns),
                    ("Categorical",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining the preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column = "math_score"
            numerical_columns = ["reading_score","writing_score"]
        
            input_feature_for_train = train_df.drop(columns=target_column)
            target_feature_for_train = train_df[target_column]

            input_feature_for_test = test_df.drop(columns=target_column)
            target_feature_for_test = test_df[target_column]

            logging.info("Apply preprocessing object to train and test dataframes")

            input_feature_for_train_arr = preprocessing_obj.fit_transform(input_feature_for_train)
            input_feature_for_test_arr = preprocessing_obj.transform(input_feature_for_test)

            train_arr = np.c_[
                input_feature_for_train_arr ,np.array(target_feature_for_train)
            ]

            test_arr = np.c_[
                input_feature_for_test_arr,np.array(target_feature_for_test)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
    