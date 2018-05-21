# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ![](/files/mjohns/housing-prices/3_ml_train_save.png)

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

data = spark.read.table("news_djia")

# COMMAND ----------

splits = data.randomSplit([0.90, 0.10], 42)
train = splits[0].cache()
test = splits[1].cache()

# COMMAND ----------

tokenizer = RegexTokenizer().setInputCol("News").setOutputCol("tokens").setPattern("\\W+")

remover = StopWordsRemover().setInputCol("tokens").setOutputCol("stopWordFree")

counts = CountVectorizer().setInputCol("stopWordFree").setOutputCol("features").setVocabSize(1000)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
evaluator.getMetricName()

# COMMAND ----------

dtc = DecisionTreeClassifier()
dtp = Pipeline().setStages([tokenizer, remover, counts, dtc])

# COMMAND ----------

print(dtc.explainParams())

# COMMAND ----------

# MAGIC %md
# MAGIC %md 
# MAGIC #### ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) 3-fold Cross Validation
# MAGIC 
# MAGIC ![crossValidation](http://curriculum-release.s3-website-us-west-2.amazonaws.com/images/301/CrossValidation.png)

# COMMAND ----------

paramGrid = ParamGridBuilder().\
    addGrid(dtc.maxBins, [64, 128]).\
    addGrid(dtc.maxDepth, [20, 30]).\
    build()
    
crossval = CrossValidator(estimator=dtp,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3, 
                          parallelism=2)

cvModel = crossval.fit(train)

# COMMAND ----------

result = cvModel.transform(test)
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC # Most [Kaggle](https://www.kaggle.com/aaron7sun/stocknews/kernels) submissions have 
# MAGIC $$AUC \epsilon [0.50,0.57] $$

# COMMAND ----------

print "AUC %(result)s" % {"result": evaluator.evaluate(result)}

# COMMAND ----------

# MAGIC %md 
# MAGIC #### ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) Model Deployment

# COMMAND ----------

model_path = "/mnt/roy/stocknews-model"
dbutils.fs.rm(model_path, True)

model = cvModel.bestModel
model.write().overwrite().save(model_path)

# COMMAND ----------

display(dbutils.fs.ls(model_path+"/stages"))

# COMMAND ----------

# MAGIC %md 
# MAGIC #<div style="float:right"><a href="$./Inference">Inference</a> <b style="font-size: 160%; color: #1CA0C2;">&#8680;</b></div></div>