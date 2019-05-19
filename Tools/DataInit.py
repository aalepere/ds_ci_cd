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
    """

    def __init__(self, source, sep, target, ratio):
        """
        Initialize the class with all the vars required for data initialization:
        - source: file path + file name of the source
        - sep: separator (semi-column, comma, tab, ...)
        - target: the name of the column in the source file that it is the target of the
          classification
        - ratio: ratio of the split between train and test
        - Load config file for replacing categorial features
        """

        self.source = source
        self.sep = sep
        self.target = target
        self.ratio = ratio

        with open("Config/cat_to_num.json") as f:
            self.conf_dict = json.load(f)

    def run(self):
        """
        run all steps one by one
        """

        self.load_file()
        self.cat_to_num()
        self.split()

    def load_file(self):
        """
        Load the source file into a DataFrame
        """

        self.df = pd.read_csv(self.source, sep=self.sep)

    def cat_to_num(self):
        """
        Replace the categorical features to a numerical mapping
        """

        for col in self.conf_dict:
            self.df = self.df.replace({col: self.conf_dict[col]})

    def split(self):
        """
        Split the DataFrame into different elements
        """

        y = self.df[self.target]
        X = self.df.drop(columns=[self.target])

        XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=self.ratio, random_state=42)

        XTrain[target] = yTrain
        self.train_data = XTrain

        XTest[target] = yTest
        self.test_data= XTest
