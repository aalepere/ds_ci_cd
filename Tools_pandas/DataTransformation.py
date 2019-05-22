import pandas as pd
import json
from scipy.stats.mstats import winsorize


class DataTransformation:
    """
    All transformation to be applied to the data before going into the model fitting
    """

    def __init__(self, init):
        """
        Initialize the class with train & test data
        Load the config file created after analysis
        """

        self.train_data = init.train_data
        self.test_data = init.test_data
        self.target = init.target

        # Load list of transformations to be applied to each feature. This pipeline instruction were
        # generated at the back of the single feature analysis performed in the jupyter notebook.
        # Please refer to the analysis folder
        with open("Config/pipeline_instructions.json") as file:
            self.config_dict = json.load(file)

    def run(self):
        """
        Run transformation for both train_data and test_data
        """

        self.transform()
        self.transform(test=True)

    def transform(self, test=False):
        """
        Apply all transformations in chain
        """

        if test:
            self.keep_features(test)
            self.replace_missings(test)
            self.discretize(test)
        else:
            self.keep_features()
            self.replace_missings()
            self.winsorize()
            self.discretize()

    def keep_features(self, test=False):
        """
        Keep the selected features going into the model
        """

        list_features = list(self.config_dict.keys())
        list_features.append(self.target)
        self.test_data = self.test_data.copy()
        self.train_data = self.train_data.copy()
        if test:
            self.test_data = self.test_data[list_features]
        else:
            self.train_data = self.train_data[list_features]

    def replace_missings(self, test=False):
        """
        Replace missing values with a default value
        """

        if test:
            # for each feature in the test set
            for col in self.test_data.drop(columns=[self.target]):
                # check if the replace missing transformation needs to be applied
                if self.config_dict[col]["replace_missings"]["apply"]:
                    self.test_data[col] = self.test_data[col].fillna(self.config_dict[col]["replace_missings"]["value"])
        else:
            for col in self.train_data.drop(columns=[self.target]):
                if self.config_dict[col]["replace_missings"]["apply"]:
                    self.train_data[col] = self.train_data[col].fillna(
                        self.config_dict[col]["replace_missings"]["value"]
                    )

    def winsorize(self):
        """
        Winsorize, to merge outliers. Only to be applied to the training data
        """

        for col in self.train_data.drop(columns=[self.target]):
            if self.config_dict[col]["winsorize"]["apply"]:
                self.train_data[col] = winsorize(self.train_data, limits=self.config_dict[col]["winsorize"]["value"])

    def discretize(self, test=False):
        """
        Discretize a continous feature into a discrete one
        """

        if test:
            for col in self.test_data.drop(columns=[self.target]):
                if self.config_dict[col]["discretize"]["apply"]:
                    # As the discretization bins were created on the training dataset, we check if
                    # any value is outside of the bin edges and we replace with the edge value
                    self.test_data[col] = self.test_data[col].apply(
                        lambda x: self.config_dict[col]["discretize"]["value"][0]
                        if x <= self.config_dict[col]["discretize"]["value"][0]
                        else x
                    )
                    self.test_data[col] = self.test_data[col].apply(
                        lambda x: self.config_dict[col]["discretize"]["value"][-1]
                        if x >= self.config_dict[col]["discretize"]["value"][-1]
                        else x
                    )
                    self.test_data[col] = pd.cut(
                        self.test_data[col],
                        bins=self.config_dict[col]["discretize"]["value"],
                        labels=False,
                        right=True,
                        include_lowest=True,
                    )
        else:
            for col in self.train_data.drop(columns=[self.target]):
                if self.config_dict[col]["discretize"]["apply"]:
                    self.train_data[col] = self.train_data[col].apply(
                        lambda x: self.config_dict[col]["discretize"]["value"][0]
                        if x <= self.config_dict[col]["discretize"]["value"][0]
                        else x
                    )
                    self.train_data[col] = self.train_data[col].apply(
                        lambda x: self.config_dict[col]["discretize"]["value"][-1]
                        if x >= self.config_dict[col]["discretize"]["value"][-1]
                        else x
                    )
                    self.train_data[col] = pd.cut(
                        self.train_data[col],
                        bins=self.config_dict[col]["discretize"]["value"],
                        labels=False,
                        right=True,
                        include_lowest=True,
                    )
