from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, Imputer, QuantileDiscretizer, StandardScaler

sc = SparkContext("local", "Spark Pipeline")
sqlContext = SQLContext(sc)

df = sqlContext.read.csv("../data/titanic.csv", sep="\t", header=True, inferSchema=True)
train, test = df.randomSplit([0.7, 0.3], seed=12345)

mapping = sqlContext.createDataFrame([(0, "male"), (1, "female")], ["id", "category"])

indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex")
train = indexer.fit(train).transform(train)
train.show()


imputer = Imputer(inputCols=["Age", "Fare"], outputCols=["out_Age", "out_Fare"]).setStrategy("median")
train = imputer.fit(train).transform(train)
train.show()

discretizer = QuantileDiscretizer(numBuckets=4, inputCol="out_Age", outputCol="out_Age_disc")
train = discretizer.fit(train).transform(train)
train.show()




