package sparknlp.optimization

/**
 * Created by Mr.gong on 2016/8/23.
 */
import breeze.linalg.{SparseVector, Vector}
import sparknlp.features.{GenerateHashFeatures, generateAllFeatures, crfsFeature}
import sparknlp.inference.LinearCliqueTree

class CrfsGradient extends Serializable{

     def computeFeatureCounts(allFeatures:Seq[crfsFeature],data:Seq[(Char,Int)],numFeatures:Int,numLabels:Int):SparseVector[Double]={
       val featureCounts=SparseVector.zeros[Double](numFeatures)

       def linearIndex(row: Int, col: Int)=row-1 + numLabels * (col-1)

       for{
         crfsFeature(nodes,assignments,idx)<-allFeatures
         if data(nodes)._2==assignments
       } featureCounts(idx)+=1
       (data.indices.map(x=>data(x)._2) zip data.indices.tail.map(x=>data(x)._2)) foreach (x=>featureCounts(linearIndex(x._1,x._2))+=1)

       featureCounts
     }

    def compute(data:Seq[(Char,Int)],weights:Vector[Double],numLabels:Int,unigram:Set[Seq[Int]]):(Vector[Double],Double)={
      val numFeatures=weights.length
      val allFeatures:Seq[crfsFeature]=GenerateHashFeatures.run(data,numLabels,unigram,numFeatures)
      val featureCounts=computeFeatureCounts(allFeatures,data,numFeatures,numLabels)
      val cliqueTree=new LinearCliqueTree(data.length,numLabels)
      cliqueTree.addFeatures(allFeatures,weights)
      val calibratedTree=cliqueTree.sumProductBP()
      val ll=calibratedTree.logZ-(featureCounts dot weights)
      val bsv=calibratedTree.computeGradient(allFeatures,weights.length,numLabels)
      (bsv-featureCounts,ll)
    }

    def compute(data:Seq[(Char,Int)],weights:Vector[Double],numLabels:Int,unigram:Set[Seq[Int]],cumGradient:Vector[Double]):Double={
      val numFeatures=weights.length
      val allFeatures:Seq[crfsFeature]=GenerateHashFeatures.run(data,numLabels,unigram,numFeatures)
      val featureCounts=computeFeatureCounts(allFeatures,data,numFeatures,numLabels)
      val cliqueTree=new LinearCliqueTree(data.length,numLabels)
      cliqueTree.addFeatures(allFeatures,weights)
      val calibratedTree=cliqueTree.sumProductBP()
      val ll=calibratedTree.logZ-(featureCounts dot weights)
      val bsv=calibratedTree.computeGradient(allFeatures,weights.length,numLabels)
      cumGradient:+=(bsv-featureCounts)
      ll
    }

     def compute(data:Seq[(Char,Int)],weights:Vector[Double],numLabels:Int,unigram:Set[Seq[Int]],conditionalFeaturesMap:Map[String,Int]):(Vector[Double],Double)={
       val numFeatures=weights.length
       val allFeatures:Seq[crfsFeature]=generateAllFeatures.run(data,numLabels,unigram,conditionalFeaturesMap)
       val featureCounts=computeFeatureCounts(allFeatures,data,numFeatures,numLabels)
       val cliqueTree=new LinearCliqueTree(data.length,numLabels)
       cliqueTree.addFeatures(allFeatures,weights)
       val calibratedTree=cliqueTree.sumProductBP()
       val ll=calibratedTree.logZ-(featureCounts dot weights)
       val bsv=calibratedTree.computeGradient(allFeatures,weights.length,numLabels)
       (bsv-featureCounts,ll)
     }

    def compute(data:Seq[(Char,Int)],weights:Vector[Double],numLabels:Int,unigram:Set[Seq[Int]],conditionalFeaturesMap:Map[String,Int],cumGradient:Vector[Double]):Double={
      val numFeatures=weights.length
      val allFeatures:Seq[crfsFeature]=generateAllFeatures.run(data,numLabels,unigram,conditionalFeaturesMap)
      val featureCounts=computeFeatureCounts(allFeatures,data,numFeatures,numLabels)
      val cliqueTree=new LinearCliqueTree(data.length,numLabels)
      cliqueTree.addFeatures(allFeatures,weights)
      val calibratedTree=cliqueTree.sumProductBP()
      val ll=calibratedTree.logZ-(featureCounts dot weights)
      val bsv=calibratedTree.computeGradient(allFeatures,weights.length,numLabels)
      cumGradient:+=(bsv-featureCounts)
      ll
    }

    def compute(data:(Vector[Double],Seq[crfsFeature],Int),weights:Vector[Double],numLabels:Int):(Vector[Double],Double)={
        val cliqueTree=new LinearCliqueTree(data._3,numLabels)
        cliqueTree.addFeatures(data._2,weights)
        val calibratedTree=cliqueTree.sumProductBP()
        val ll=calibratedTree.logZ-(data._1 dot weights)
        val bsv=calibratedTree.computeGradient(data._2,weights.length,numLabels)
       (bsv-data._1,ll)
    }

    def compute(data:(Vector[Double],Seq[crfsFeature],Int),weights:Vector[Double],numLabels:Int,cumGradient:Vector[Double]):Double={
      val cliqueTree=new LinearCliqueTree(data._3,numLabels)
      cliqueTree.addFeatures(data._2,weights)
      val calibratedTree=cliqueTree.sumProductBP()
      val bsv=calibratedTree.computeGradient(data._2,weights.length,numLabels)
      val ll=calibratedTree.logZ-(data._1 dot weights)
      cumGradient:+=(bsv-data._1)
      ll
    }

}
