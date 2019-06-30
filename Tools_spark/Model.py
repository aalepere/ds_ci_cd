from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler


class Model:
    """
    This class instantiate a logic regression with the transformed data sets; to then fit and make
    predictions with the test data
    """

    def __init__(self, trans):
        """
        Iniatialize the dataset used and the algorithm to be used
        """

        self.train_data = trans.train_data
        self.test_data = trans.test_data

    def run(self):
        """
        run in sequence, fit and then test
        """

        self.fit()
        self.test()

    def fit(self):
        """
        fit the logistic regression
        """

        # Convert all features into a vectors call features
        assembler = VectorAssembler(
            inputCols=["Age_replace_missings", "Fare_discretized", "Sex_indexed"], outputCol="features"
        )
        self.output = assembler.transform(self.train_data)
        lr = LogisticRegression(featuresCol="features", labelCol="Survived", maxIter=10)
        self.lrModel = lr.fit(self.output)

    def test(self):
        """
        predict agains test set and retrieves confusion matrix
        """

        trainingSummary = self.lrModel.summary
        roc = trainingSummary.roc.toPandas()
        print("Gini: " + str(2 * trainingSummary.areaUnderROC - 1))
