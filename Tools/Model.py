from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import metrics


class Model:
    """
    XXX
    """

    def __init__(self, trans):
        """
        XXX
        """

        self.XTrain = trans.train_data.drop(columns=[trans.target])
        self.yTrain = trans.train_data[trans.target]

        self.XTest = trans.test_data.drop(columns=[trans.target])
        self.yTest = trans.test_data[trans.target]

        self.logisticRegr = LogisticRegression(solver="liblinear")

    def run(self):
        """
        XXX
        """

        self.fit()
        self.test()

    def fit(self):
        """
        XXX
        """

        self.logisticRegr.fit(self.XTrain, self.yTrain)

    def test(self):
        """
        XXX
        """

        predictions = self.logisticRegr.predict(self.XTest)
        cm = metrics.confusion_matrix(self.yTest, predictions)
        print(cm)
