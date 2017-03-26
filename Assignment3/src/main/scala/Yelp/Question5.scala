package Yelp

/**
  * Created by dheerajgoudborlla on 08-03-2017.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object Question5
{
  def main(args: Array[String])
  {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Question2").setMaster("local[*]")
    val sc = new SparkContext(conf)
    if (args.length != 2)
    {
      println("Missing Arguments or Incorrect Usage")
      sys.exit()
    }

    val business = sc.textFile(args(0))
    val review = sc.textFile(args(1))

    val bSplit = business.map(line => line.split('^'))
    val rSplit = review.map(line => line.split('^'))

    val bData = bSplit.map(line => (line(0), line(1))).filter(line => line._2.contains("TX"))

    val rData1 = rSplit.map(line => (line(2), 1)).reduceByKey(_ + _)

    val rData2 = review.map(line => line.split('^')).map(line => (line(2), line(3).toDouble)).reduceByKey(_+_)

    val rData = rData1.join(rData2)

    val result = bData.join(rData).map(line => (line._1, (line._2._2._1, line._2._2._2))).distinct()

    //result.saveAsTextFile(args(2))
    result.collect().foreach(println)
    sys.exit()
  }
}
