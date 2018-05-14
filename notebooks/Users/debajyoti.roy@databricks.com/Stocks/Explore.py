# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

path = "/mnt/roy/kaggle/stocknews"
display(dbutils.fs.ls(path))

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

stocks_w_label = stocks.\
  withColumn("window_values", collect_list(col("Close")).over(window_spec)).\
  filter(size(col("window_values")) == 2).\
  withColumn("label", get_label(col("window_values"))).\
  orderBy(asc("source"), asc("date"))

display(stocks_w_label)

# COMMAND ----------

news = spark.read.csv(path+"/RedditNews.csv", header=True)
news.printSchema()

# COMMAND ----------

news.withColumn("Date", to_date('Date', 'yyyy-MM-dd'))
display(news)

# COMMAND ----------

data = stocks_w_label.\
  join(news, "Date").\
  withColumn("News", regexp_replace("News", "b'", "")).\
  withColumn("News", regexp_replace("News", "[^\w\s]", "")).\
  select("Date", "label", lower(col("News")).alias("News")).\
  orderBy(asc("Date"))
  
display(data)

# COMMAND ----------

data.createOrReplaceTempView("news_djia")

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESC news_djia

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT min(Date), max(Date) FROM news_djia

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT count(1), label FROM news_djia GROUP BY label

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT length(News) from news_djia

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT label, count(News) FROM news_djia WHERE News LIKE '%brexit%' GROUP BY label

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT label, count(News) FROM news_djia WHERE News LIKE '%gdp%' GROUP BY label