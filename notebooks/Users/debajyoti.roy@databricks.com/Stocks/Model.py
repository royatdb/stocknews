# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ![](/files/mjohns/housing-prices/3_ml_train_save.png)

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

data = spark.read.table("news_djia")

# COMMAND ----------

splits = data.randomSplit([0.8, 0.2], 42)
train = splits[0].cache()
test = splits[1].cache()

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

tokenizer = RegexTokenizer()    \
  .setInputCol("News")        \
  .setOutputCol("tokens")       \
  .setPattern("\\W+")

remover = StopWordsRemover()    \
  .setInputCol("tokens")        \
  .setOutputCol("stopWordFree") \

counts = CountVectorizer()      \
  .setInputCol("stopWordFree")  \
  .setOutputCol("features")     \
  .setVocabSize(1000)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
evaluator.getMetricName()

# COMMAND ----------

dtc = DecisionTreeClassifier()
dtp = Pipeline().setStages([tokenizer, remover, counts, dtc])

# COMMAND ----------

print(dtc.explainParams())

# COMMAND ----------

paramGrid = ParamGridBuilder().\
    addGrid(dtc.maxBins, [32, 64]).\
    addGrid(dtc.maxDepth, [5, 10]).\
    addGrid(dtc.impurity, ["gini", "entropy"]).\
    build()
    
crossval = CrossValidator(estimator=dtp,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

cvModel = crossval.fit(train)

# COMMAND ----------

result = cvModel.transform(test)
display(result)

# COMMAND ----------

print "AUC %(result)s" % {"result": evaluator.evaluate(result)}

# COMMAND ----------

model_path = "/mnt/roy/stocknews-model"
model = cvModel.bestModel
model.write().overwrite().save(model_path)

# COMMAND ----------

display(dbutils.fs.ls(model_path+"/stages"))

# COMMAND ----------

# MAGIC %md 
# MAGIC #<div style="float:right"><a href="$./Inference">Inference</a> <b style="font-size: 160%; color: #1CA0C2;">&#8680;</b></div></div>

# COMMAND ----------

