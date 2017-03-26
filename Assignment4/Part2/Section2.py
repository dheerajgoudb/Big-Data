# Databricks notebook source
# MAGIC %md
# MAGIC ### ***PART-III (Section 2: Clustering)***
# MAGIC In this section, we cluster the dataset into K different clusters using clustering techniques like 'K-Means' Clustering.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ***User Knowledge Modeling Data Set***
# MAGIC Abstract: It is the real dataset about the students' knowledge status about the subject of Electrical DC Machines. http://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling

# COMMAND ----------

students = sc.textFile("/FileStore/tables/6rsya5vv1490134063282/StudentKnowledgeData.csv")

# COMMAND ----------

#In the above loaded file, the first row contain the names of the columns. We remove it and parse the dataset
colNames = students.first()
studentsRDD = students.filter(lambda x: x != colNames).map(lambda line: [float(i) for i in line.split(',')])
print 'Number of Rows: %s' %studentsRDD.count()
print 'First two rows: %s' %studentsRDD.take(2)

# COMMAND ----------

#Now we will load the packages
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt

#Build the model
clusters = KMeans.train(studentsRDD, 4, maxIterations = 10,initializationMode = "random")

#calculating sum of squared errors
def error(point):
  center = clusters.centers[clusters.predict(point)]
  return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = studentsRDD.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Sum of Squared Error = " + str(WSSSE))

# COMMAND ----------

#Now, we will calculate Sum of Squared Error for different values of k
for k in range(1,5):
  clusters = KMeans.train(studentsRDD, k, maxIterations = 10,initializationMode = "random")
  WSSSE = studentsRDD.map(lambda point: error(point)).reduce(lambda x, y: x + y)
  print("Sum of Squared Error for "+ str(k) +"= " + str(WSSSE))
