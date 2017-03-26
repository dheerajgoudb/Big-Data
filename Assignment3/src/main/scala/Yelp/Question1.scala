package Yelp

/**
  * Created by dheerajgoudborlla on 06-03-2017.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object Question1
{
  def main(args: Array[String])
  {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Question1").setMaster("local[*]")
    val sc = new SparkContext(conf)

    if (args.length != 2)
      {
        println("Missing arguments")
       sys.exit()
      }

    val business = sc.textFile(args(0))
    val review = sc.textFile(args(1))

    val bSplit = business.map { x => x.split('^') }
    val rSplit=review.map { x => x.split('^') }

    val b1 = bSplit.map ( line => ( line(0),line(1))).distinct
    val b2 = bSplit.map ( line => ( line(0),line(2))).distinct

    val r1 = rSplit.map(line => (line(2), 1 )).reduceByKey(_+_)
    val r2 = rSplit.map(line => (line(2), line(3).toDouble )).reduceByKey(_+_)

    val r = r1.join(r2)

    val rFinal = r.map(a=> ( a._1, a._2._2/a._2._1))

    val bFinal = b1.join(b2)

    val Final = bFinal.join(rFinal)

    val result = Final.map(line => (line._1, (line._2._1._1, line._2._1._2, line._2._2)))

    val list = result.collect()

    val topRated = list.sortWith(_._2._3 > _._2._3)

    //val topRDD = sc.parallelize(topRated.take(10))

    //topRDD.saveAsTextFile(args(2))
    topRated.take(10).foreach(println)
    sys.exit()
  }
}
