import math
import json
from pyspark.ml.feature import Bucketizer, Imputer
from pyspark.sql.functions import expr


class DataTransformation:
    """
    All transformation to be applied to the data before going into the model fitting
    """

    def __init__(self, init):
        """
        Initialize the class with train & test data
        Load the config file created after analysis
        """

        self.train_data = init.train
        self.test_data = init.test
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

        list_features = ['Age','Sex_indexed','Fare']
        list_features.append(self.target)
        if test:
            self.test_data = self.test_data.select(*list_features)
        else:
            self.train_data = self.train_data.select(*list_features)

    def replace_missings(self, test=False):
        """
        Replace missing values with a default value
        """

        for col in list(self.config_dict.keys()):
            # check if the replace missing transformation needs to be applied
            if self.config_dict[col]["replace_missings"]["apply"]:
                imputer = Imputer(inputCols=[col], outputCols=["{}_replace_missings".format(col)]).setMissingValue(
                    self.config_dict[col]["replace_missings"]["value"]
                )
                if test:
                    self.test_data = imputer.fit(self.test_data).transform(self.test_data)
                else:
                    self.train_data = imputer.fit(self.train_data).transform(self.train_data)

    def winsorize(self):
        """
        Winsorize, to merge outliers. Only to be applied to the training data
        """

        for col in list(self.config_dict.keys()):
            # check if the replace missing transformation needs to be applied
            if self.config_dict[col]["winsorize"]["apply"]:
                lower = self.config_dict[col]["winsorize"]["value"]
                higher = 1 - self.config_dict[col]["winsorize"]["value"]
                percentiles = self.train_data.approxQuantile("{}".format(col), [lower, higher], lower)
                winsorize = expr(
                    """IF({} >= {}, {},IF({} <= {},{},{}))""".format(
                        col, percentiles[0], percentiles[0], col, percentiles[1], percentiles[1], col
                    )
                )
                self.train_data.withColumn("{}".format(col), winsorize)

    def discretize(self, test=False):
        """
        Discretize a continous feature into a discrete one
        """

        for col in list(self.config_dict.keys()):
            # check if the discretizer transformation needs to be applied
            if self.config_dict[col]["discretize"]["apply"]:
                splits = self.config_dict[col]["discretize"]["value"]
                splits = [-math.inf] + splits
                splits = splits + [math.inf]
                bucketizer = Bucketizer(
                    splits=splits,
                    inputCol=col,
                    outputCol="{}_discretized".format(col),
                )
                if test:
                    self.test_data = bucketizer.transform(self.test_data)
                else:
                    self.train_data = bucketizer.transform(self.train_data)
