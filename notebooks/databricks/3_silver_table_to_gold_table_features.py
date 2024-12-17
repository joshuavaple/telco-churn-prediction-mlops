# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Load Silver, preview, define types of cols

# COMMAND ----------

df_silver = spark.read.table("corp_nonprod.silver.ops_telcochurn_tab_churn")

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog `corp_nonprod`; select * from `silver`.`ops_telcochurn_tab_churn` limit 100;

# COMMAND ----------

# defining types of columns: ID, label, numerical features, categorical features
label_col = 'churn'
id_col = 'customerid'

num_feats = [c[0] for c in df_silver.dtypes if c[1] in ['int', 'double']]
num_feats.remove(label_col)
cat_feats = [c[0] for c in df_silver.dtypes if c[1] == 'string']
cat_feats.remove(id_col)


# COMMAND ----------

num_feats

# COMMAND ----------

cat_feats

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Splitting with stratification

# COMMAND ----------

from pyspark.sql.functions import col, lit
from pyspark.sql import DataFrame

def stratified_split(df, label_col, train_ratio, seed):
    fractions = df.select(label_col).distinct().withColumn("fraction", lit(train_ratio)).collect()
    fractions_dict = {row[label_col]: row["fraction"] for row in fractions}
    train_df = df.sampleBy(label_col, fractions_dict, seed)
    test_df = df.subtract(train_df)
    return train_df, test_df

# COMMAND ----------

df_silver_train, df_silver_test = stratified_split(df=df_silver, label_col=label_col, train_ratio=0.8, seed=42)

# COMMAND ----------

# compare the label value distribution in the train and test set:
# convert to percentage:
df_silver_train.groupBy("churn").count().withColumn("percentage", col("count")/df_silver_train.count()).show()
df_silver_test.groupBy("churn").count().withColumn("percentage", col("count")/df_silver_test.count()).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit transfoming pipeline to training data, and transform both training and testing data

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# COMMAND ----------

stages = []
for col in cat_feats:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid='keep')
    stages.append(indexer)

    encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded")
    stages.append(encoder)

assembler = VectorAssembler(inputCols=num_feats, outputCol="num_features", handleInvalid='keep')
stages.append(assembler)
scaler = StandardScaler(inputCol="num_features", outputCol="scaled_num_features", withMean=True, withStd=True)
stages.append(scaler)

# Create a pipeline with the stages
pipeline = Pipeline(stages=stages)


# COMMAND ----------

# Fit the pipeline on the training data
pipeline_model = pipeline.fit(df_silver_train)

# COMMAND ----------

# Transform the training data
df_silver_train_encoded = pipeline_model.transform(df_silver_train)

# Transform the test data using the same pipeline model
df_silver_test_encoded = pipeline_model.transform(df_silver_test)

# COMMAND ----------

# Display the encoded DataFrames
display(df_silver_train_encoded)
display(df_silver_test_encoded)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train a model

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Define the feature columns and target column
feature_cols = [col for col in df_silver_train_encoded.columns if col.endswith('_encoded') or col.endswith('_features')]
target_col = 'churn'

# Assemble the feature columns into a single vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="all_features")
df_silver_train_assembled = assembler.transform(df_silver_train_encoded)
df_silver_test_assembled = assembler.transform(df_silver_test_encoded)


# COMMAND ----------

# Select the features and target columns
train_data = df_silver_train_assembled.select("all_features", target_col)
test_data = df_silver_test_assembled.select("all_features", target_col)

# COMMAND ----------

# train_data.display(5)
# check if churn has any NaN:
train_data.filter(train_data.all_features.isNull()).count()

# COMMAND ----------

# check if any NaN in the features:
train_data.filter(train_data.all_features.isna()).count()

# COMMAND ----------

# Train an XGBoost model on the training data
# xgb_classifier = SparkXGBClassifier(label_col=target_col, features_col="all_features")
# xgb_model = xgb_classifier.fit(train_data)

# from pyspark.ml.classification import RandomForestClassifier
# rf_classifier = RandomForestClassifier(labelCol=target_col, featuresCol="all_features")
# rf_model = rf_classifier.fit(train_data)

from pyspark.ml.classification import LogisticRegression
lr_classifier = LogisticRegression(labelCol=target_col, featuresCol="all_features")
lr_model = lr_classifier.fit(train_data)
