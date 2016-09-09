package sparknlp.features

import breeze.linalg.SparseVector

/**
 * Created by Mr.gong on 2016/8/20.
 */

case class crfsFeature(nodes:Int,assignment:Int,idx:Int)

class generateAllFeatures {
  def run(
           sentence:Seq[(Char,Int)],
           numLabels:Int,
           unigram:Set[Seq[Int]],
           conditionalFeaturesMap:Map[String,Int]):Seq[crfsFeature]={
          // val len=sentence.length
         //  val PairsUnconditonalFeatures=generatePairsUnconditonalFeatures(numLabels)
           val preMaxFeatures=numLabels*numLabels
           val conditionalFeaturs= generateConditionalFeaturs(sentence,conditionalFeaturesMap,preMaxFeatures,unigram,numLabels)
    conditionalFeaturs
  }

  //private def sub2Ind(majorStride:Int)(row: Int, col: Int)=row + majorStride * col

  private  def generateConditionalFeaturs(
                                           sentence:Seq[(Char,Int)],
                                           conditionalFeaturesMap:Map[String,Int],
                                           preMaxFeatures:Int,
                                           unigram:Set[Seq[Int]],
                                           numLabels:Int):Seq[crfsFeature]={
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
             featuresMapIndx=conditionalFeaturesMap.get(str)
             if featuresMapIndx.isDefined
           }  yield{
             crfsFeature(index,label,featuresMapIndx.get+preMaxFeatures)
           }
  }

 /* private  def generatePairsUnconditonalFeatures(numLabels:Int):Seq[crfsFeature]={

           def linearIndex(row: Int, col: Int)=row-1 + numLabels * (col-1)

           for {
             row<-1 to numLabels
             col<-1 to numLabels
           } yield {
             crfsUncontionalPairFeature(Seq(row,col),linearIndex(row,col))
           }
  }*/

}

object generateAllFeatures{
  def run(
           sentence:Seq[(Char,Int)],
           numLabels:Int,
           unigram:Set[Seq[Int]],
           conditionalFeaturesMap:Map[String,Int]):Seq[crfsFeature]=
  new generateAllFeatures().run(
    sentence:Seq[(Char,Int)],
    numLabels:Int,
    unigram:Set[Seq[Int]],
    conditionalFeaturesMap:Map[String,Int])
}
