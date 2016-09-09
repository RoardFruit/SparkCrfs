package sparknlp.inference

import breeze.linalg.{sum, SparseVector, argmax, max}
import breeze.numerics.{exp, log}
import sparknlp.features.crfsFeature

/**
 * Created by Mr.gong on 2016/8/23.
 */
class LinearCalibratedCliqueTree(val calibratedCliques:Seq[twoFactor]){

      def computeMaxMarginalsBP():Seq[Int]={
          val tail=argmax((calibratedCliques.last maxMarginalization calibratedCliques.length-1).value)+1
          calibratedCliques.indices.map{
            idx=>
            argmax((calibratedCliques(idx) maxMarginalization(idx+1)).value)+1
          }:+tail
      }

      def computeMarginalsBP():Seq[singleFactor]={
          val tail=calibratedCliques.last marginalization calibratedCliques.length-1 normalize()
        calibratedCliques.indices.map{
          idx=>
            calibratedCliques(idx).marginalization(idx+1).normalize()
        }:+tail
      }

      def computeGradient(features:Seq[crfsFeature],numFeatures:Int,numLabels:Int):SparseVector[Double]={
        val sigleMarginal=computeMarginalsBP()
        val nomalCalibratedCliques=calibratedCliques.map(_.normalize())
        val BSV=SparseVector.zeros[Double](numFeatures)
        features foreach{
          case crfsFeature(nodes,assignment,idx)=>
              val featureExpectedValue:Double=sigleMarginal(nodes).value(assignment-1)
              BSV(idx)+=featureExpectedValue
        }

       def linearIndex(row: Int, col: Int)=row-1 + numLabels * (col-1)

       for{
         nomalCalibratedClique<-nomalCalibratedCliques
         row<-1 to numLabels
         col<-1 to numLabels
       }  BSV(linearIndex(row, col))+=nomalCalibratedClique.value(row-1,col-1)

        BSV
      }

      def logZ:Double= {
        val pi_max = max(calibratedCliques.head.value)
        pi_max+log(sum(exp(calibratedCliques.head.value-pi_max)))
      }
}
