package sparknlp.features

/**
 * Created by Mr.gong on 2016/8/19.
 */
import breeze.linalg._
import org.apache.spark.rdd.RDD

//need to change,too large
class FeaturesTransform {
    def run(
            input:RDD[Seq[(Char,Int)]],
            numLabels:Int,
            unigram:Set[Seq[Int]],
            minCount:Int):(RDD[(SparseVector[Double],Seq[crfsFeature],Int)],Int)={
    val featuresMap=GenerateConditonalFeaturesMap.run(input,numLabels,unigram,minCount)
    val numFeatures=featuresMap.size+numLabels*numLabels
      (for{
      sentence<-input
    } yield {
      val crfsFeatures=generateAllFeatures.run(sentence,numLabels,unigram,featuresMap)
      val featureCounts=SparseVector.zeros[Double](numFeatures)
      for{
        crfsFeature(nodes,assignments,idx)<-crfsFeatures
        if sentence(nodes)._2==assignments
      } featureCounts(idx)+=1

      def linearIndex(row: Int, col: Int)=row-1 + numLabels * (col-1)

       (sentence.indices.map(x=>sentence(x)._2) zip sentence.indices.tail.map(x=>sentence(x)._2)) foreach (x=>featureCounts(linearIndex(x._1,x._2))+=1)

      (featureCounts,crfsFeatures,sentence.length)
    },numFeatures)
  }
}

object FeaturesTransform {
  def run(
           input:RDD[Seq[(Char,Int)]],
           numLabels:Int,
           unigram:Set[Seq[Int]],
           minCount:Int):(RDD[(SparseVector[Double],Seq[crfsFeature],Int)],Int)=
  new FeaturesTransform().run(
    input:RDD[Seq[(Char,Int)]],
    numLabels:Int,
    unigram:Set[Seq[Int]],
    minCount:Int)
}

