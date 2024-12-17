from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

spark = SparkSession.builder.getOrCreate()

schema = StructType([
    StructField('CustomerID', StringType(), True),
    StructField('FirstName', StringType(), True),
    StructField('LastName', StringType(), True),
])

data = [
    ['1', 'John', 'Doe'],
    ['2', 'Jane', 'Doe'],
    ['3', 'John', 'Smith'],
]

customers = spark.createDataFrame(data, schema)
customers.show()