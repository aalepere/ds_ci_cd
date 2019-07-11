from pickle import load

from flask import Flask
from flask_restful import Api, Resource, reqparse
from Tools_pandas.DataInit import DataInit
from Tools_pandas.DataTransformation import DataTransformation

app = Flask(__name__)
api = Api(app)

# Definition of fields provided through body of the POST REQUEST
parser = reqparse.RequestParser()
parser.add_argument("Sex")
parser.add_argument("Age", type=int)
parser.add_argument("Fare", type=int)

# Unpickle model
with open("model.p", "rb") as f:
    model = load(f)

class Model(Resource):
    """
    Returns the probability of survival based on inputs posted via API
    """

    def post(self):
        # Parse arguments from body request
        args = parser.parse_args()
        # Create dictionary based on arguments
        source = {"Sex": args["Sex"], "Age": args["Age"], "Fare": args["Fare"]}
        # Initialize - mainly converting string features to numeric ones
        init = DataInit(source, "Survived", 1, prod=True)
        init.run()
        # Transform information provided
        trans = DataTransformation(init, prod=True)
        trans.run()
        # Run model
        response = {}
        proba = model.predict_proba(trans.test_data)
        response.update({"probability_of_survival": proba[:,1][0]})
        return response, 201


# Enable model endpoint
api.add_resource(Model, "/model")

if __name__ == "__main__":
    app.run()
