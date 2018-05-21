// Databricks notebook source
// MAGIC %md
// MAGIC ![](/files/mjohns/housing-prices/4_ml_deploy.png)

// COMMAND ----------

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

import org.apache.spark.sql.functions._

// COMMAND ----------

val model = PipelineModel.load("/mnt/roy/stocknews-model")

// COMMAND ----------

val tree = model.stages.last.asInstanceOf[DecisionTreeClassificationModel]
display(tree)

// COMMAND ----------

// MAGIC %md 
// MAGIC #### ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) Inference on Stream of News

// COMMAND ----------

val newsDF = spark.read.
  option("header", "true").
  option("inferSchema", "true").
  csv("mnt/roy/kaggle/stocknews/RedditNews.csv").
  withColumn("Date", to_date($"Date")).
  withColumn("News", regexp_replace($"News", "b'", "")).
  withColumn("News", regexp_replace($"News", "[^\\w\\s]", "")).
  withColumn("News", lower($"News")).
  groupBy($"Date").
  agg(concat_ws(" ", collect_list($"News")).alias("News"))

val newsSchema = newsDF.schema

val newsJsonPath = "/mnt/roy/redditnews"
dbutils.fs.rm(newsJsonPath, true)

newsDF.coalesce(64).write.json(newsJsonPath)

val newsStream = spark.readStream.option("maxFilesPerTrigger",1).schema(newsSchema).json(newsJsonPath)

// COMMAND ----------

display(newsStream)

// COMMAND ----------

val predictedStream = model.transform(newsStream)

display(predictedStream)

// COMMAND ----------

// MAGIC %md 
// MAGIC #### ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) Backtesting
// MAGIC Strategy: If model predicts 1 then, `buy` at start of the day and `sell` at close

// COMMAND ----------

val actuals = spark.read.table("news_djia")

display(
  predictedStream
    .join(actuals, "Date")
    .withColumn("year", year($"Date"))
    .select("year", "close_movement", "prediction")
    .filter("prediction = 1")
    .groupBy("year")
    .agg(sum("close_movement"))
    .orderBy("year")
)

// COMMAND ----------

// MAGIC %md 
// MAGIC #### ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) TL;DR 
// MAGIC ![ml](https://s3.us-east-2.amazonaws.com/databricks-roy/MLDB.jpeg)