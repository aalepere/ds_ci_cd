import json

from pyspark.ml.feature import StringIndexer


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

    def __init__(self, sqlContext, source, sep, target, ratio):
        """
        Initialize the class with all the vars required for data initialization:
        - sqlContext: Spark SQL context to be set out of the class
        - source: file path + file name of the source
        - sep: separator (semi-column, comma, tab, ...)
        - target: the name of the column in the source file that it is the target of the
          classification
        - ratio: ratio of the split between train and test
        - Load config file for replacing categorial features
        """

        # Spark SQL contexts
        self.sqlContext = sqlContext

        self.source = source
        self.sep = sep
        self.target = target
        self.ratio = ratio

        # Load the mapping dictionary that will be used to replace categorical features into
        # numerical values, note that for SPARK it will be used only to know which features do
        # apply for such transformation
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

        self.df = self.sqlContext.read.csv(self.source, sep=self.sep, header=True, inferSchema=True)

    def cat_to_num(self):
        """
        Replace the categorical features to numerical values based on the mapping provided
        """

        for col in self.conf_dict:
            indexer = StringIndexer(inputCol=col, outputCol="{}_indexed".format(col))
            self.df = indexer.fit(self.df).transform(self.df)

    def split(self):
        """
        Random split of train and test of the inital dataset; based on the ration provided at the
        instanciation
        """

        ratio_c = 1 - self.ratio
        self.train, self.test = self.df.randomSplit([self.ratio, ratio_c], seed=12345)
