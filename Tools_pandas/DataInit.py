import json

import pandas as pd
from sklearn.model_selection import train_test_split


class DataInit:
    """
    Initialize the data that will be used for the model:
    - Load the source data
    - Split the data into different elements:
        - XTrain: data that will be used for training
        - yTrain: target of the training
        - XTest: data that will be used for testing
        - yTest: target of the test
    - prod flag; it to apply specific steps once model has been promoted to production environment
    """

    def __init__(self, source, target, ratio, sep=None, prod=False):
        """
        Initialize the class with all the vars required for data initialization:
        - source: file path + file name of the source
        - sep: separator (semi-column, comma, tab, ...)
        - target: the name of the column in the source file that it is the target of the
          classification
        - ratio: ratio of the split between train and test
        - prod: to be used only for production
        - Load config file for replacing categorial features
        """

        self.source = source
        self.sep = sep
        self.target = target
        self.ratio = ratio
        self.prod = prod
        self.train_data = None

        # Load the mapping dictionary that will be used to replace categorical features into
        # numerical values
        with open("../Config/cat_to_num.json") as f:
            self.conf_dict = json.load(f)

    def run(self):
        """
        run all steps one by one
        """

        self.load_file()
        self.cat_to_num()
        if not self.prod:
            self.split()  # Split between train and test only required during dev phase
        if self.prod:
            self.test_data = self.df

    def load_file(self):
        """
        Load the source file into a DataFrame
        """

        if self.prod:
            # For prod use dictionary derived from the body of the POST Request
            self.df = pd.DataFrame([self.source])
        else:
            self.df = pd.read_csv(self.source, sep=self.sep)

    def cat_to_num(self):
        """
        Replace the categorical features to numerical values based on the mapping provided
        """

        for col in self.conf_dict:
            self.df = self.df.replace({col: self.conf_dict[col]})

    def split(self):
        """
        Random split of train and test of the inital dataset; based on the ration provided at the
        instanciation
        """

        # Define the target column we want to predict
        y = self.df[self.target]
        X = self.df.drop(columns=[self.target])

        XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=self.ratio, random_state=42)

        # Append back to targer column to both train and test; this is to ensure that after all the
        # transformation applied we still have the correct target column in front of the
        # transformed features
        XTrain = XTrain.copy()
        XTrain.loc[:, self.target] = yTrain
        self.train_data = XTrain

        XTest = XTest.copy()
        XTest.loc[:, self.target] = yTest
        self.test_data = XTest
