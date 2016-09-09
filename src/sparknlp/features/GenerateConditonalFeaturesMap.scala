package sparknlp.features

/**
 * Created by Mr.gong on 2016/8/19.
 */
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging

class GenerateConditonalFeaturesMap extends Logging{

  def featureCount(input: RDD[Seq[(Char,Int)]],unigram:Set[Seq[Int]]):Seq[(String,Int)]={
        val featureCountRDD=
        for{
          sentence<- input
          index<- sentence.indices
          (offsets,numUnigram)<-unigram.zipWithIndex
        } yield {
          val str='U'+numUnigram.toString+
            offsets.map{offset=>
            val temp=offset+index
            if(temp<0) "_B"+temp.toString else
              if(temp>=sentence.length) "_B"+(temp-sentence.length+1).toString else sentence(temp)._1
          }.mkString("/")+sentence(index)._2.toString
          str->1
        }
        featureCountRDD.reduceByKey(_+_).collect()
    }

  def run(input: RDD[Seq[(Char,Int)]],unigram:Set[Seq[Int]],minCount:Int):Map[String,Int]=
      featureCount(input,unigram:Set[Seq[Int]]).withFilter(x=>x._2>=minCount).map(x=>x._1).zipWithIndex.toMap
}

object GenerateConditonalFeaturesMap{

  def featureCount(input: RDD[Seq[(Char,Int)]],numLabels:Int,unigram:Set[Seq[Int]])=
    new GenerateConditonalFeaturesMap().featureCount(input: RDD[Seq[(Char,Int)]],unigram:Set[Seq[Int]])

  def run(input: RDD[Seq[(Char,Int)]],numLabels:Int,unigram:Set[Seq[Int]],minCount:Int)=
      new GenerateConditonalFeaturesMap().run(input: RDD[Seq[(Char,Int)]],unigram:Set[Seq[Int]],minCount:Int)

}
