# Databricks notebook source
# MAGIC %md
# MAGIC ![](/files/mjohns/housing-prices/1_data_engineering.png)

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/royatdb/stocknews/master/DJIA_table.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/royatdb/stocknews/master/RedditNews.csv

# COMMAND ----------

# MAGIC %md 
# MAGIC #### ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) Data Ingest

# COMMAND ----------

dbutils.fs.rm("dbfs:/mnt/roy/kaggle/stocknews",True)
dbutils.fs.mkdirs("dbfs:/mnt/roy/kaggle/stocknews")

# COMMAND ----------

# MAGIC %sh
# MAGIC cp *.csv /dbfs/mnt/roy/kaggle/stocknews

# COMMAND ----------

display(
  dbutils.fs.ls("/mnt/roy/kaggle/stocknews")
)

# COMMAND ----------

# MAGIC %md 
# MAGIC #<div style="float:right"><a href="$./Explore">Explore</a> <b style="font-size: 160%; color: #1CA0C2;">&#8680;</b></div></div>