# Databricks notebook source
# MAGIC %md
# MAGIC # Reading a connected storage account

# COMMAND ----------

STRG_PATH = "abfss://datascientist@psaigdapadlsuat001.dfs.core.windows.net/dfe/Joshuale/telco_churn_prediction/telecom_churn_with_id.csv"
df = spark.read.csv(STRG_PATH, header=False, inferSchema=True)
df.show()

# COMMAND ----------

# by right, we should not see anything outside of our project folder, will resolve later
dbutils.fs.ls("abfss://datascientist@psaigdapadlsuat001.dfs.core.windows.net/dfe/Joshuale/telco_churn_prediction/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Listing

# COMMAND ----------

# using spark.sqk, we can perform typical SQL queries in a python-based medium
# list all catalogs that you can access:
spark.sql("SHOW CATALOGS").show()

# list all schemas in a catalog:
spark.sql("SHOW SCHEMAS IN corp_nonprod").show()

# list all tables in a schema:
spark.sql("SHOW TABLES IN corp_nonprod.bronze").show()

# list all volumes in a schema:
spark.sql("SHOW VOLUMES IN databricks_telco_customer_dataset.v01").show()

# COMMAND ----------

# List all files in a specific catalog volume
files = dbutils.fs.ls("/Volumes/databricks_telco_customer_dataset/v01/telco/")
for file in files:
    print(file.path)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Reading data objs with `spark.read.`

# COMMAND ----------

# MAGIC %md
# MAGIC ### a. Read a file from a volume into a spark df

# COMMAND ----------

# read file content in a volume into a spark df:
PATH = "/Volumes/databricks_telco_customer_dataset/v01/telco/telco-customer-churn.csv"
df = spark.read.format("csv").load(path=PATH,header=True, inferSchema=True)

# COMMAND ----------

# we can access the schema of the spark df above
df.schema

# COMMAND ----------

# we can define a schema to avoid errors in the inferred schema:
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

schema = StructType([
    StructField('customerID', StringType(), True),
    StructField('gender', StringType(), True),
    StructField('SeniorCitizen', IntegerType(), True),
    StructField('Partner', StringType(), True),
    StructField('Dependents', StringType(), True),
    StructField('tenure', IntegerType(), True),
    StructField('PhoneService', StringType(), True),
    StructField('MultipleLines', StringType(), True),
    StructField('InternetService', StringType(), True),
    StructField('OnlineSecurity', StringType(), True),
    StructField('OnlineBackup', StringType(), True),
    StructField('DeviceProtection', StringType(), True),
    StructField('TechSupport', StringType(), True),
    StructField('StreamingTV', StringType(), True),
    StructField('StreamingMovies', StringType(), True),
    StructField('Contract', StringType(), True),
    StructField('PaperlessBilling', StringType(), True),
    StructField('PaymentMethod', StringType(), True),
    StructField('MonthlyCharges', DoubleType(), True),
    StructField('TotalCharges', StringType(), True),
    StructField('Churn', StringType(), True)
])

# load the data with the defined schema
df = spark.read.format("csv").schema(schema).load(path=PATH,header=True, inferSchema=False)
display(df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### b. Read a table in a schema

# COMMAND ----------

# loading a table to a spark df
# syntax: spark.read.table("catalog.schema.table")
df_bronze = spark.read.table("corp_nonprod.bronze.telco_customer_churn")
display(df_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC ### c. Spark dataframe operations

# COMMAND ----------

# GET DATAFRAME DIMENSIONS
print(df_bronze.count())
print(len(df_bronze.columns))

# COMMAND ----------

# SINGLE ROW CONDITION
# get all rows where SeniorCitizen = 1:
# in Pandas: df_senior = df_bronze[df_bronze['SeniorCitizen']==1]
df_senior = df_bronze.filter(df_bronze['SeniorCitizen']==1)
display(df_senior.limit(5))

# COMMAND ----------

# showing a dataframe with native pyspark method
# however, this is in raw text and not pretty in a notebook environment
# use the databricks display() function above
df_senior.show(2, truncate=10)

# COMMAND ----------

# MULTIPLE ROW CONDITIONS
# get all rows where SeniorCitizen = 1 and Churn = 'Yes':
df_senior_churn = df_bronze.filter((df_bronze['SeniorCitizen'] == 1) & (df_bronze['Churn'] == 'Yes'))
display(df_senior_churn.limit(5))

# COMMAND ----------

# SELECT A COLUMN (SERIES)
col = df_bronze.select('Churn') # this is still a df
print(type(col)) 
display(col.limit(5))

# COMMAND ----------

col = df_bronze['Churn']
print(type(col))
# we cannot use limit on a col object, use the method above with select instead
try:
  display(col.limit(5))
except Exception as e:
  print(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Other Useful Codes

# COMMAND ----------

# check the table details with spark.sql
catalog_name = "corp_nonprod"
table_name = "telco_customer_churn"
schema_name = "bronze"

details = spark.sql(f"DESCRIBE DETAIL {catalog_name}.{schema_name}.{table_name}")
display(details)

# COMMAND ----------

# checking table details using catalog may be restricted in the workspace
try:
  table_details = spark.catalog.getTable(f"{catalog_name}.{schema_name}.{table_name}")
except Exception as e:
  print(e)
