# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ![](/files/mjohns/housing-prices/3_ml_train_save.png)

# COMMAND ----------

from pyspark.ml.feature import IDF, Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, NGram, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

data = spark.read.table("news_djia")

# COMMAND ----------

splits = data.randomSplit([0.85, 0.15], 42)
train = splits[0].cache()
test = splits[1].cache()

# COMMAND ----------

tokenizer = [RegexTokenizer().setInputCol("News").setOutputCol("tokens").setPattern("\\W+")]

remover = [StopWordsRemover().setInputCol("tokens").setOutputCol("stopWordFree")]
  
n=2

ngrams = [
    NGram(n=i, inputCol="stopWordFree", outputCol="{0}_grams".format(i))
    for i in range(1, n + 1)
]

cv = [
    CountVectorizer(vocabSize=1000,inputCol="{0}_grams".format(i),
        outputCol="{0}_tf".format(i))
    for i in range(1, n + 1)
]

idf = [IDF(inputCol="{0}_tf".format(i), outputCol="{0}_tfidf".format(i), minDocFreq=5) for i in range(1, n + 1)]

assembler = [VectorAssembler(
    inputCols=["{0}_tfidf".format(i) for i in range(1, n + 1)],
    outputCol="features"
)]

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
evaluator.getMetricName()

# COMMAND ----------

dtc = DecisionTreeClassifier()
dtp = Pipeline(stages = tokenizer+remover+ngrams+cv+idf+assembler+[dtc] )

# COMMAND ----------

print(dtc.explainParams())

# COMMAND ----------

# MAGIC %md
# MAGIC ![crossValidation](http://curriculum-release.s3-website-us-west-2.amazonaws.com/images/301/CrossValidation.png)

# COMMAND ----------

paramGrid = ParamGridBuilder().\
    addGrid(dtc.maxBins, [32, 64]).\
    addGrid(dtc.maxDepth, [5, 30]).\
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
dbutils.fs.rm(model_path, True)

model = cvModel.bestModel
model.write().overwrite().save(model_path)

# COMMAND ----------

display(dbutils.fs.ls(model_path+"/stages"))

# COMMAND ----------

# MAGIC %md 
# MAGIC #<div style="float:right"><a href="$./Inference">Inference</a> <b style="font-size: 160%; color: #1CA0C2;">&#8680;</b></div></div>