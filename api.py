from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Model(Resource):
    """
    Returns the probability of survival based on inputs posted via API
    """

    def post(self):
        args = parser.parse_args()
        sex = args['sex']
        age = args['age']
        fare = args['fare']
        #TODO add model
        return "XXX", 201

api.add_resource(Model, '/model')

if __name__ == '__main__':
    app.run(debug=True)
