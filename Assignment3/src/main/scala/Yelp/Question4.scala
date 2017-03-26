package Yelp

/**
  * Created by dheerajgoudborlla on 08-03-2017.
  */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object Question4
{
  def main(args: Array[String])
  {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Question4").setMaster("local[*]")
    val sc = new SparkContext(conf)
    if (args.length != 2)
      {
        println("Missing Arguments")
        sys.exit()
      }

    val review = sc.textFile(args(0))
    val user = sc.textFile(args(1))

    val rSplit = review.map(line => line.split('^'))
    val uSplit = user.map(line => line.split('^'))

    val rData = rSplit.map(line => (line(1),1)).reduceByKey(_+_)

    val uData = uSplit.map(line => (line(0),line(1)))

    val ruData = rData.join(uData)

    val CollectedData = ruData.collect()

    val sortd = CollectedData.sortWith(_._2._1> _._2._1).take(10)

    val finalResult=sortd.map(line => ( line._1,line._2._2))

    //val finalRDD = sc.parallelize(finalResult)

    //finalRDD.saveAsTextFile(args(2))
    finalResult.foreach(println)
    sys.exit()
  }
}
