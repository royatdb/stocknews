# Databricks notebook source
# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/royatdb/stocknews/master/DJIA_table.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/royatdb/stocknews/master/RedditNews.csv

# COMMAND ----------

# MAGIC %sh ls -lh .

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