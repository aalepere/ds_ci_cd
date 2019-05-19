import pandas as pd


class Model:
    """
    XXX
    """

    def __init__(self, tran):
        """
        XXX
        """

        self.XTrain = trans.train_data.drop(columns=[trans.target])
        self.yTrain = trans.train_data[trans.target]

        self.XTest = trans.test_data.drop(columns=[trans.target])
        self.yTest = trans.test_data[trans.target]

    def fit():
        """
        XXX
        """

        pass

    def test(self):
        """
        XXX
        """

        pass
