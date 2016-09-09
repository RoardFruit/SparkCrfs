package sparknlp.inference

import breeze.linalg.{SparseVector, DenseMatrix}
import breeze.numerics.exp
import sparknlp.features.{ crfsFeature}
import breeze.linalg.Vector
/**
 * Created by Mr.gong on 2016/8/22.
 */
trait CliqueTree {
  import LinearCliqueTree._
      val cliques:Seq[Factor]

      def addFeatures(features:Seq[crfsFeature],weights:Vector[Double])

      def calibrate(union:(twoFactor,singleFactor)=>twoFactor,out:(twoFactor,Int)=>singleFactor):LinearCalibratedCliqueTree

      def sumProductBP():LinearCalibratedCliqueTree=calibrate(productFactor,marginalization)

      def maxSumBP():LinearCalibratedCliqueTree=calibrate(sumFactor,maxMarginalization)

}


class LinearCliqueTree(linearLength:Int,numLabels:Int) extends CliqueTree{
      val cliques:Seq[twoFactor]=
        for{
           idx<-0 until linearLength-1
        } yield twoFactor(Seq(idx,idx+1),DenseMatrix.zeros[Double](numLabels,numLabels))

      def addFeatures(crfsFeatures:Seq[crfsFeature],weights:Vector[Double])={
        /*val feature2cliqueIdx:Seq[Int]=
            for{
              feature<-features
            } yield {
              if(feature.nodes.length==1) cliques.indexWhere(factor=>factor.nodes.contains(feature.nodes.head))
              else cliques.indexWhere(factor=>factor.nodes==feature.nodes)
            }

        features zip feature2cliqueIdx foreach{
          case (feature,idx)=> {
            if (idx == -1) {println(feature);cliques foreach(x=>println(x.nodes));println(linearLength)}
            cliques(idx) addFeature(feature, weights(feature.idx))
          }
          }*/
        crfsFeatures foreach(feature=>{
          val idx=if (feature.nodes==cliques.length) feature.nodes-1 else feature.nodes
          cliques(idx) addFeature(feature,weights(feature.idx))
        }
          )

        def linearIndex(row: Int, col: Int)=row-1 + numLabels * (col-1)

        for{
          clique<-cliques
          row<-1 to numLabels
          col<-1 to numLabels
        } clique addPairUncontionalFeature(row-1,col-1,weights(linearIndex(row,col)))

        }


     /* def calibrate(calibrateWay:Int):LinearCalibratedCliqueTree={

        val forwardMessageHead:singleFactor=
          if(calibrateWay==0) cliques.head marginalization 0  else cliques.head maxMarginalization 0

        val forwardMessage:Seq[singleFactor]=
          {if(calibrateWay==0) (1 until linearLength-2).scanLeft(forwardMessageHead)((message,cliqueIdx)=>cliques(cliqueIdx) productFactor message marginalization cliqueIdx)
          else (1 until linearLength-2).scanLeft(forwardMessageHead)((message,cliqueIdx)=>cliques(cliqueIdx) sumFactor message maxMarginalization cliqueIdx)}

        val backwardMessageLast=
          if(calibrateWay==0) cliques.last marginalization linearLength-1 else cliques.last maxMarginalization linearLength-1

        val backwardMessage=
        {if(calibrateWay==0) (1 until linearLength-2).scanRight(backwardMessageLast)((cliqueIdx,message)=>cliques(cliqueIdx) productFactor message marginalization cliqueIdx+1)
         else (1 until linearLength-2).scanRight(backwardMessageLast)((cliqueIdx,message)=>cliques(cliqueIdx) sumFactor message maxMarginalization cliqueIdx+1)}

        val calibratedCliques:Seq[twoFactor]=(0 until linearLength-1) map{
          case 0=>
            if(calibrateWay==0) cliques(0) productFactor backwardMessage(0) else cliques(0) sumFactor backwardMessage(0)
          case idx if idx==linearLength-2=>
            if(calibrateWay==0) cliques(idx) productFactor forwardMessage.last else cliques(idx) sumFactor forwardMessage.last
          case idx=>
            if(calibrateWay==0) cliques(idx) productFactor forwardMessage(idx-1) productFactor backwardMessage(idx)
            else cliques(idx) sumFactor forwardMessage(idx-1) sumFactor backwardMessage(idx)
        }

        new LinearCalibratedCliqueTree(calibratedCliques)
      }*/

      def calibrate(union:(twoFactor,singleFactor)=>twoFactor,out:(twoFactor,Int)=>singleFactor):LinearCalibratedCliqueTree={

        val forwardMessageHead:singleFactor=
          out(cliques.head,0)

        val forwardMessage:Seq[singleFactor]=
        (1 until linearLength-2).scanLeft(forwardMessageHead)((message,cliqueIdx)=>out(union(cliques(cliqueIdx), message),cliqueIdx))

        val backwardMessageLast=
         out(cliques.last,linearLength-1)

        val backwardMessage=
        (1 until linearLength-2).scanRight(backwardMessageLast)((cliqueIdx,message)=>out(union(cliques(cliqueIdx),message),cliqueIdx+1))


        val calibratedCliques:Seq[twoFactor]=(0 until linearLength-1) map{
          case 0=>
            union(cliques(0),backwardMessage(0))
          case idx if idx==linearLength-2=>
            union(cliques(idx),forwardMessage.last)
          case idx=>
            union(union(cliques(idx),forwardMessage(idx-1)),backwardMessage(idx))
        }

        new LinearCalibratedCliqueTree(calibratedCliques)
  }
}

object LinearCliqueTree{
  def productFactor(twofactor:twoFactor,singlefactor:singleFactor):twoFactor=twofactor productFactor singlefactor

  def sumFactor(twofactor:twoFactor,singlefactor:singleFactor):twoFactor=twofactor productFactor singlefactor

  def marginalization(twofactor:twoFactor,outVar:Int)=twofactor marginalization outVar

  def maxMarginalization(twofactor:twoFactor,outVar:Int)=twofactor maxMarginalization outVar
}