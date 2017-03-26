package Yelp

/**
  * Created by dheerajgoudborlla on 08-03-2017.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object Question2
{
  def main(args: Array[String])
  {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Question2").setMaster("local[*]")
    val sc = new SparkContext(conf)
    if (args.length != 4)
      {
        println("Missing Arguments")
       sys.exit()
      }

    val review = sc.textFile(args(0))
    val user = sc.textFile(args(1))
    val firstname = args(2)
    val lastname = args(3)
    val fullname = firstname + " " + lastname

    val rSplit = review.map(line => line.split('^'))
    val uSplit = user.map(line => line.split('^'))

    val rData = rSplit.map(line => (line(1), line(3)))
    val uData = uSplit.map(line => (line(0), line(1))).distinct()

    val urData = uData.join(rData)

    val req = urData.map(line => (line._2._1, line._2._2.toDouble))

    val userFilter = req.filter(_._1.contains(fullname))

    val total = userFilter.count()

    val result = userFilter.reduceByKey(_+_).map(line => (line._1, line._2 / total))

    //result.saveAsTextFile(args(4))
    result.collect().foreach(println)
    sys.exit()
  }
}
