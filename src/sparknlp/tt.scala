package sparknlp

/**
 * Created by Mr.gong on 2016/8/19.
 */

import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkConf, SparkContext}
import breeze.linalg._
import sparknlp.features.{crfsFeature, FeaturesTransform, generateAllFeatures, GenerateConditonalFeaturesMap}
import breeze.stats.mean
import sparknlp.inference.LinearCliqueTree
import sparknlp.optimization._
import sparknlp.sparkcrfs.{SparkCrfsHashModel, SparkCrfsModelWithSGD, SparkCrfsModel, SparkCrfsWithSGD}

import scala.io.Source

object tt extends App {
  val conf = new SparkConf().setAppName("tt").setMaster("spark://master:7077").set("spark.executor.memory","6g")
  val sc = new SparkContext(conf)
  sc.addJar("/home/hadoop/sparkcrfs4/out/artifacts/sparkcrfs/sparkcrfs.jar")
  val u = Set(Seq(-1),Seq(0),Seq(1),Seq(-1,0,1),Seq(-1,0),Seq(0,1))
  val c:RDD[String]= sc.textFile("hdfs://master:9000/icwb2-data/training/msr_training_chu",24)

 val trainRDD:RDD[Seq[(Char,Int)]]=c.filter(_.nonEmpty).map{
   x=>
   val sentence=x.split(" ")
   sentence.map{
     y=>
       (y(0),y(1).toInt-48)
   }
 }
val tRDD=trainRDD.filter(x=>x.length>1).repartition(24).cache()
private val gradient=new CrfsGradient
private val updater= new SimpleUpdater()
/*val optimizer= new crfsGradientDescent(gradient,updater,u)
  .setNumIterations(1000)
  .setMiniBatchFraction(0.1)
  .setNumLabers(4)
val weights = optimizer.optimize(tRDD)*/
// val model = SparkCrfsWithSGD.train(tRDD, 100, 4,0.001,1,u)
//tRDD.map(x=>model.predict(x,4,u)) foreach println
val (weights,loss) = LBFGSforCRFsWithHash.runLBFGS(
tRDD,
4,
u,
gradient,
updater,
10,
1e-4,
200,
0,
1<<19)
val model=new SparkCrfsHashModel(weights)
  val p:RDD[String]= sc.textFile("hdfs://master:9000/icwb2-data//gold/msr_test_chu",24)
  val ptrainRDD:RDD[Seq[(Char,Int)]]=p.filter(_.nonEmpty).map{
    x=>
      val sentence=x.split(" ")
      sentence.map{
        y=>
          (y(0),y(1).toInt-48)
      }
  }
  val ptRDD=ptrainRDD.filter(x=>x.length>1).cache()

 val (cor,tol)=model.predict(ptRDD,4,u).flatMap(x=>x).map(x=>(if(x._1._2==x._2) 1 else 0,1)).reduce((x,y)=>(x._1+y._1,x._2+y._2))
 println(cor/tol.toDouble)
//  val nonsize=weights.toArray.filter(_>0).size
}