package Yelp

/**
  * Created by dheerajgoudborlla on 08-03-2017.
  */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object Question3
{
  def main(args: Array[String])
  {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Question3").setMaster("local[*]")
    val sc = new SparkContext(conf)
    if (args.length != 2)
      {
        println("Missing Arguments")
        sys.exit()
      }

    val business = sc.textFile(args(0))
    val review = sc.textFile(args(1))

    val bSplit = business.map(line => line.split('^'))
    val rSplit = review.map(line => line.split('^'))

    val Bdata = bSplit.map(line => (line(0), line(1))).filter(line => line._2.contains("Stanford"))

    val Rdata = rSplit.map(line => (line(2), (line(1), line(3))))

    val finalData = Bdata.join(Rdata).map(line => (line._2._2._1, line._2._2._2))

    //finalData.saveAsTextFile(args(2))
    finalData.collect().foreach(println)
    sys.exit()
  }
}
