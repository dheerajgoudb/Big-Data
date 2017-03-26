# Databricks notebook source
# MAGIC %md
# MAGIC #### **Parsing the Dataset**
# MAGIC We read in each of the files and create an RDD consisting of parsed lines.
# MAGIC Each line in the ratings dataset (`ratings.dat`) is formatted as:
# MAGIC   `UserID,ProfileID,Rating`
# MAGIC Each line in the movies (`gender.dat`) dataset is formatted as:
# MAGIC   `UserID,Gender`
# MAGIC The format of these files is uniform and simple, so we can use Python [`split()`](https://docs.python.org/2/library/stdtypes.html#str.split) to parse their lines.
# MAGIC Parsing the two files yields two RDDS
# MAGIC * For each line in the ratings dataset, we create a tuple of (UserID, ProfileID, Rating).
# MAGIC * For each line in the gender dataset, we create a tuple of (UserID, Gender).

# COMMAND ----------

ratings = sc.textFile("/FileStore/tables/ve4iplo41489724819120/ratings.dat")
genders = sc.textFile("/FileStore/tables/ve4iplo41489724819120/gender.dat")

def get_ratings_tuple(line):
  ''' Parse a line in the movies dataset
  Args:
        entry (str): a line in the ratings dataset in the form of UserID, ProfileID, Rating
  Returns:
        tuple: (UserID, ProfileID, Rating)
  '''
  line = line.split(',')
  return int(line[0]), int(line[1]), int(line[2])

def get_genders_tuple(line):
  ''' Parse a line in the movies dataset
  Args:
        entry (str): a line in the ratings dataset in the form of UserID, Gender
  Returns:
        tuple: (UserID, Gender)
  '''
  line = line.split(',')
  return int(line[0]), line[1]

ratingsRDD = ratings.map(lambda line: get_ratings_tuple(line)).cache()
gendersRDD = genders.map(lambda line: get_genders_tuple(line)).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC #### ***Top 10 Profiles***
# MAGIC Now, I output the top 10 profileID which have the highest average rating.

# COMMAND ----------

#First, we should transform ratingsRDD to another RDD of form (ProfileID, (Rating1, Rating2, Rating3,..))
profileIDandRatingsTuple = ratingsRDD.map(lambda x: (x[1], x[2])).groupByKey()

#The 'profileIDandRatingsTuple' RDD is transformed into another RDD of form (ProfileRDD, avgRating).
getProfileAvgRatings = profileIDandRatingsTuple.map(lambda x: (x[0], float(sum(x[1])) / len(x[1])))

#Now we will output the top 10 profile's with the highest average rating
top10 = getProfileAvgRatings.takeOrdered(10, key = lambda x: -x[1])
print "The top 10 highest rated Profiles are: %s\n" %top10

# COMMAND ----------

# MAGIC %md
# MAGIC #### ***Training and Testing Data***
# MAGIC Next the dataset is divided into training and testing data using randomSplit() function.

# COMMAND ----------

trainingRDD, testRDD, validationRDD = ratingsRDD.randomSplit([6,2,2], seed = 0L)
print "Training: %s" %trainingRDD.count()
print "Testing: %s" %testRDD.count()
print "Validation: %s" %validationRDD.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### ***RMSE***
# MAGIC Now we will define a function to compute RMSE between actual and predicted values

# COMMAND ----------

import math

def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                      (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda line: ((line[0], line[1]), line[2]))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda line: ((line[0], line[1]), line[2]))

    # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
    # RDD) in the reformatted RDDs using RDD transformtions - do not use collect()
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD).map(lambda line: (line[1][1] - line[1][0])**2))

    # Compute the total squared error - do not use collect()
    totalError = squaredErrorsRDD.reduce(lambda x,y: x + y)

    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return math.sqrt(totalError * 1.0 / numRatings)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ***Training an ALS Model***

# COMMAND ----------

# MAGIC %md
# MAGIC First we use the validationRDD and try different parametres like different rank and find which has the low error

# COMMAND ----------

from pyspark.mllib.recommendation import ALS

#creating an RDD of form (UserID, ProfileID)
validationForPredictRDD = validationRDD.map(lambda x: (x[0], x[1]))

seed = 5L
iterations = 5
regularizationParameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.03

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank

# COMMAND ----------

myModel = ALS.train(trainingRDD, bestRank, seed=seed, iterations=iterations, lambda_=regularizationParameter)

testForPredictedRDD = testRDD.map(lambda x: (x[0], x[1]))

predictedTestRDD = myModel.predictAll(testForPredictedRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

# COMMAND ----------

print 'The model had a RMSE on the test set of %s' % testRMSE
