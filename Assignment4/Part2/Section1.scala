// Databricks notebook source
//Importing all the required packages
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

// COMMAND ----------

// MAGIC %md
// MAGIC ### ***I. Classification***
// MAGIC Dataset: Balance Scale Data Set (http://archive.ics.uci.edu/ml/datasets/Balance+Scale)

// COMMAND ----------

val balance = sc.textFile("/FileStore/tables/4sbztyuq1490545604931/balance_scale-3159c.data")

//Defining a class
def getDoubleValue( input:String ) : Double = {
    var result:Double = 0.0
    if (input == "B")  result = 0.0 
    if (input == "L")  result = 1.0
    if (input == "R")  result = 2.0
    return result
   }

//creating a class
case class Balance(label:Int, features:Vector)

val balanceDF = balance.map({ line =>
  val x = line.split(',')
  Balance(getDoubleValue(x(0)).toInt, Vectors.dense(x(1).toDouble, x(2).toDouble, x(3).toDouble, x(4).toDouble))
}).toDF()

// COMMAND ----------

balanceDF.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ### ***1. First Attempt***
// MAGIC Including all the features to build the model

// COMMAND ----------

// MAGIC %md
// MAGIC ##### ***DECISION TREE CLASSIFIER***

// COMMAND ----------

val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(balanceDF)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(balanceDF)

//dividing dataset into testing and training data
val Array(trainingData, testData) = balanceDF.randomSplit(Array(0.8, 0.2), seed = 1L)

val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//creating a pipeline model
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

//model is trained with the training dataset
val model = pipeline.fit(trainingData)
//Model is used to predict the test data
val predictions = model.transform(testData)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

// COMMAND ----------

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### ***Evaluation Metrics Calculation***

// COMMAND ----------

//extracting only (predictedlabels, labels)
val predictionsAndLabels = predictions.rdd.map(x => (x(7).toString().toDouble, x(0).toString().toDouble))

//Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionsAndLabels)

// AUPRC
val auPRC = metrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)

// ROC Curve
val roc = metrics.roc

// AUROC
val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### ***RANDOM FOREST CLASSIFIER***

// COMMAND ----------

val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(balanceDF)

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(balanceDF)

val Array(trainingData, testData) = balanceDF.randomSplit(Array(0.8, 0.2))

val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

// COMMAND ----------

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### ***Evaluation Metrics Calculation***

// COMMAND ----------

//extracting only (predictedlabels, labels)
val predictionsAndLabels = predictions.rdd.map(x => (x(7).toString().toDouble, x(0).toString().toDouble))

//Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionsAndLabels)

// AUPRC
val auPRC = metrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)

// ROC Curve
val roc = metrics.roc

// AUROC
val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

// COMMAND ----------

// MAGIC %md
// MAGIC ### ***2. PCA***
// MAGIC Second Attempt: Performing Dimensionality Reduction

// COMMAND ----------

import org.apache.spark.ml.feature.PCA
import org.apache.spark.sql.functions.col

val dimReducePCA = new PCA().setInputCol("features").setK(1).fit(balanceDF).setOutputCol("pcaFeatures")
val pcaDF = dimReducePCA.transform(balanceDF).drop(col("features"))

// COMMAND ----------

// MAGIC %md
// MAGIC ##### ***I. PCA - DECISION TREE CLASSIFIER***

// COMMAND ----------

val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(pcaDF)
val featureIndexer = new VectorIndexer().setInputCol("pcaFeatures").setOutputCol("indexedFeatures").setMaxCategories(4).fit(pcaDF)

val Array(trainingData, testData) = pcaDF.randomSplit(Array(0.8, 0.2), seed = 1L)

val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

val model = pipeline.fit(trainingData)
val predictions = model.transform(testData)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

// COMMAND ----------

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------

//extracting only (predictedlabels, labels)
val predictionsAndLabels = predictions.rdd.map(x => (x(7).toString().toDouble, x(0).toString().toDouble))

//Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionsAndLabels)

// AUPRC
val auPRC = metrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)

// ROC Curve
val roc = metrics.roc

// AUROC
val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### ***II. PCA - RANDOM FOREST CLASSIFIER***

// COMMAND ----------

val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(pcaDF)

val featureIndexer = new VectorIndexer().setInputCol("pcaFeatures").setOutputCol("indexedFeatures").setMaxCategories(4).fit(pcaDF)

val Array(trainingData, testData) = pcaDF.randomSplit(Array(0.8, 0.2))

val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

// COMMAND ----------

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------

//extracting only (predictedlabels, labels)
val predictionsAndLabels = predictions.rdd.map(x => (x(7).toString().toDouble, x(0).toString().toDouble))

//Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionsAndLabels)

// AUPRC
val auPRC = metrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)

// ROC Curve
val roc = metrics.roc

// AUROC
val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)
