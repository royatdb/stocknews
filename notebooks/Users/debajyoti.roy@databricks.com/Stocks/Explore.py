# Databricks notebook source
# MAGIC %md
# MAGIC ![](/files/mjohns/housing-prices/2_data_science.png)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

path = "/mnt/roy/kaggle/stocknews"
display(dbutils.fs.ls(path))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) Data Manipulation

# COMMAND ----------

raw_stocks = spark.read.csv(path+"/DJIA_table.csv", header=True)
raw_stocks.printSchema()

# COMMAND ----------

stocks = raw_stocks.select("Date", "Close").\
  withColumn("Date", to_date("Date")).\
  withColumn("Close", raw_stocks['Close'].cast(DoubleType())).\
  withColumn("source", lit("DJIA")).\
  orderBy(asc("Date"))

display(stocks)

# COMMAND ----------

window_spec = Window.partitionBy("source").orderBy(asc("Date")).rowsBetween((-1), 0)

get_label = udf(lambda closes: 1 if closes[1]>closes[0] else 0, IntegerType())

get_close_movement = udf(lambda closes: closes[1] - closes[0], DoubleType())

stocks_w_label = stocks.\
  withColumn("window_values", collect_list(col("Close")).over(window_spec)).\
  filter(size(col("window_values")) == 2).\
  withColumn("label", get_label(col("window_values"))).\
  withColumn("close_movement", get_close_movement(col("window_values"))).\
  orderBy(asc("source"), asc("date"))

display(stocks_w_label)

# COMMAND ----------

raw_news = spark.read.csv(path+"/RedditNews.csv", header=True)
raw_news.printSchema()

# COMMAND ----------

news = raw_news.\
  withColumn("Date", to_date(col("Date"))).\
  withColumn("News", regexp_replace("News", "b'", "")).\
  withColumn("News", regexp_replace("News", "[^\w\s]", "")).\
  withColumn("News", lower(col("News"))).\
  groupBy(col("Date")).\
  agg(concat_ws(" ", collect_list(col("News"))).alias("News"))
  
display(news)

# COMMAND ----------

news.printSchema()

# COMMAND ----------

data = stocks_w_label.\
  join(news, "Date").\
  select("Date", "label", "close_movement", "News").\
  orderBy(asc("Date"))
  
display(data)

# COMMAND ----------

data.write.mode("overwrite").saveAsTable("news_djia")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) Data Exploration

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESC news_djia

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT min(Date), max(Date) FROM news_djia

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT count(1) from news_djia

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT count(1), label FROM news_djia GROUP BY label

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT length(News) from news_djia

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT label, count(News) FROM news_djia WHERE News LIKE '%sanctions%' GROUP BY label

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT label, count(News) FROM news_djia WHERE News LIKE '%olympics%' GROUP BY label

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT close_movement FROM news_djia

# COMMAND ----------

# MAGIC %md 
# MAGIC #<div style="float:right"><a href="$./Model">Model</a> <b style="font-size: 160%; color: #1CA0C2;">&#8680;</b></div></div>