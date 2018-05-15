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

val newsDeltaPath = "/mnt/roy/redditnews"
dbutils.fs.rm(newsDeltaPath, true)

newsDF.write.format("delta").save(newsDeltaPath)

val newsStream = spark.readStream.option("maxFilesPerTrigger",1).format("delta").load(newsDeltaPath)

// COMMAND ----------

val predictedStream = model.transform(newsStream)
display(predictedStream)

// COMMAND ----------

val predictedStream = model.transform(newsStream)
display(predictedStream.groupBy("prediction").count())

// COMMAND ----------

// MAGIC %md #TL;DR 
// MAGIC ![ml](https://s3.us-east-2.amazonaws.com/databricks-roy/MLDB.jpeg)

// COMMAND ----------

