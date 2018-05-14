# Databricks notebook source
path = "/mnt/roy/kaggle/stocknews"
display(dbutils.fs.ls(path))

# COMMAND ----------

stocks = spark.read.csv(path+"/DJIA_table.csv", header=True, inferSchema=True)
display(stocks)

# COMMAND ----------

news = spark.read.csv(path+"/RedditNews.csv", header=True, inferSchema=True)
display(news)

# COMMAND ----------

