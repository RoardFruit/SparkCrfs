package sparknlp.features

import org.apache.spark.mllib.feature.HashingTF

/**
 * Created by Mr.gong on 2016/9/8.
 */
class GenerateHashFeatures {
  def run(
           sentence:Seq[(Char,Int)],
           numLabels:Int,
           unigram:Set[Seq[Int]],
           numFeatures:Int):Seq[crfsFeature]={
    // val len=sentence.length
    //  val PairsUnconditonalFeatures=generatePairsUnconditonalFeatures(numLabels)
    val preMaxFeatures=numLabels*numLabels
    val conditionalFeaturs= generateConditionalHashFeaturs(sentence,preMaxFeatures,unigram,numLabels,numFeatures)
    conditionalFeaturs
  }


  private  def generateConditionalHashFeaturs(
                                           sentence:Seq[(Char,Int)],
                                           preMaxFeatures:Int,
                                           unigram:Set[Seq[Int]],
                                           numLabels:Int,
                                           numFeatures:Int):Seq[crfsFeature]={
    val hashingTF = new HashingTF(numFeatures)
    for {
      label<- 1 to numLabels
      index<- sentence.indices
      (offsets,numUnigram)<-unigram.zipWithIndex
      str='U'+numUnigram.toString+
        offsets.map{offset=>
          val temp=offset+index
          if(temp<0) "_B"+temp.toString else
          if(temp>=sentence.length) "_B"+(temp-sentence.length+1).toString else sentence(temp)._1
        }.mkString("/")+label.toString
    }  yield{
      crfsFeature(index,label,hashingTF.indexOf(str))
    }
  }
}

object GenerateHashFeatures{
  def run(
           sentence:Seq[(Char,Int)],
           numLabels:Int,
           unigram:Set[Seq[Int]],
           numFeatures:Int):Seq[crfsFeature]=
    new GenerateHashFeatures().run(
      sentence,
      numLabels,
      unigram,
      numFeatures)
}