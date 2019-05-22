from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import metrics


class Model:
    """
    This class instantiate a logic regression with the transformed data sets; to then fit and make
    predictions with the test data
    """

    def __init__(self, trans):
        """
        Iniatialize the dataset used and the algorithm to be used
        """

        self.XTrain = trans.train_data.drop(columns=[trans.target])
        self.yTrain = trans.train_data[trans.target]

        self.XTest = trans.test_data.drop(columns=[trans.target])
        self.yTest = trans.test_data[trans.target]

        self.logisticRegr = LogisticRegression(solver="liblinear")

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

        self.logisticRegr.fit(self.XTrain, self.yTrain)

    def test(self):
        """
        predict agains test set and retrieves confusion matrix
        """

        predictions = self.logisticRegr.predict(self.XTest)
        cm = metrics.confusion_matrix(self.yTest, predictions)
        print(cm)
