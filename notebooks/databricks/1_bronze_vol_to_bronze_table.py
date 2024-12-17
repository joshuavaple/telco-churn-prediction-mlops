# Databricks notebook source
# import os
# os.listdir("/Workspace/Users/joshuale@globalpsa.com/GitHub/telco-churn-prediction-mlops/data/bronze/")

# COMMAND ----------

# # rading from "local" files encounters [PATH_NOT_FOUND] error
# df = spark.read.csv(f"dbfs:/Workspace/Users/joshuale@globalpsa.com/GitHub/telco-churn-prediction-mlops/data/bronze/telecom_churn_with_id.csv", header=True, inferSchema=True)


# COMMAND ----------

# read a file from this catalog location: databricks_telco_customer_dataset.v01.telco.telco-customer-churn.csv
df = spark.read.format("csv").load("/Volumes/databricks_telco_customer_dataset/v01/telco/telco-customer-churn.csv",
                                   header=True, inferSchema=True)
display(df)

# COMMAND ----------



# COMMAND ----------

df.write.mode("overwrite").saveAsTable("corp_nonprod.bronze.ops_telcochurn_tab_churn")

# COMMAND ----------

# write this spark df to a table in another catalog corp_nonprod/bronze
# df.createOrReplaceTempView("new_data")

# spark.sql("""
# MERGE INTO corp_nonprod.bronze AS target
# USING new_data AS source
# ON target.customerID = source.customerID
# WHEN NOT MATCHED THEN
#   INSERT *
# """)

# COMMAND ----------

# set working directory to this active notebook's directory:
dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
