import json

class DataTransformation:
    """
    XXX
    """

    def __init__(self, train_data, test_data):
        """
        XXX
        """

        self.train_data = train_data
        self.tes_data = test_data

        with open("Config/pipeline_instructions.json") as file:
            self.config_dict = json.load(file)

    def run(self):
        """
        XXX
        """

        pass

    def transform(self, test=False):
        """
        XXX
        """

        pass

    def replace_missings(self):
        """
        XXX
        """

        pass

    def winsorize(self):
        """
        XXX
        """

        pass

    def discretize(self):
        """
        XXX
        """

        pass
