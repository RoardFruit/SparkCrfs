package sparknlp.inference

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import sparknlp.features.crfsFeature
import breeze.linalg.Vector
/**
 * Created by Mr.gong on 2016/8/21.
 */
trait Factor{

     /*def union(other:Factor,unionWay:(Double,Double)=>Double):Factor

     def product(other:Factor):Factor=union(other,_*_)

     def sum(other:Factor):Factor=union(other,_+_)

     def summedOut(vnodes:Set[Int],summedOutWay:(Double,Double)=>Double):Factor

     def marginalization(vnodes:Set[Int]):Factor=summedOut(vnodes,_+_)

     def maxMarginalization(vnodes:Set[Int]):Factor=summedOut(vnodes,(a,b)=>if (a>b) a else b)*/
}

class singleFactor(val node:Int,val value:DenseVector[Double]) extends Factor{
  //0 sum 1 max
  def union(other:twoFactor,unionWay:Int):twoFactor=other union(this,unionWay)

  def productFactor(other:twoFactor):twoFactor=union(other,0)

  def sumFactor(other:twoFactor):twoFactor=union(other,1)

  def normalize():singleFactor={
    val pi_max = max(value)
    val logsumexp=pi_max+log(sum(exp(value-pi_max)))
    singleFactor(node,exp(value-logsumexp))
  }

}

object singleFactor{
  def apply(node:Int,value:DenseVector[Double])=
    new singleFactor(node,value)
}


class twoFactor(val nodes:Seq[Int],val value:DenseMatrix[Double]) extends Factor{

     require(nodes.length==2,"twoFactor requires size 2")
     //0 product 1 sum
     def normalize():twoFactor={
       val pi_max = max(value)
       val logsumexp=pi_max+log(sum(exp(value-pi_max)))
       twoFactor(nodes,exp(value-logsumexp))
     }

     def union(other:singleFactor,unionWay:Int):twoFactor={
         require(nodes.contains(other.node),"union need sharenodes")
         val newvalue:DenseMatrix[Double]=
           if (nodes.head==other.node) {
               value(::,*) :+ other.value
           }
           else{
             value(*,::) :+ other.value
           }
         twoFactor(nodes,newvalue)
      }

     def productFactor(other:singleFactor):twoFactor=union(other,0)

     def sumFactor(other:singleFactor):twoFactor=union(other,1)

     //0 sum 1 max
     def summedOut(node:Int,summedOutWay:Int):singleFactor={
       require(nodes.contains(node),"summedOut need sharenodes")
       val newvalue:DenseVector[Double]=
         if (nodes.head==node) {
           if (summedOutWay==0) {
             val pi_max=max(value.t(*,::))
             pi_max+log(sum(exp(value.t(::, *)-pi_max),Axis._1))
           } else max(value.t(*,::))
         }
         else{
           if (summedOutWay==0) {
             val pi_max=max(value(*,::))
             pi_max+log(sum(exp(value(::, *)-pi_max),Axis._1))
           } else max(value(*,::))
         }
       if (nodes.head==node) singleFactor(nodes.last,newvalue)
       else singleFactor(nodes.head,newvalue)
     }

     def marginalization(node:Int):singleFactor=summedOut(node,0)

     def maxMarginalization(node:Int):singleFactor=summedOut(node,1)

     def addFeature(feature:crfsFeature,newvalue:Double)={
           if(nodes.head==feature.nodes) value(feature.assignment-1,::):+=newvalue
           else value(::,feature.assignment-1):+=newvalue
        //  value(feature.assignment.head-1,feature.assignment(1)-1)+=newvalue
     }

     def addPairUncontionalFeature(row:Int,col:Int,newvalue:Double)=
       value(row,col)+=newvalue


}

object twoFactor{
  def apply(nodes:Seq[Int],value:DenseMatrix[Double])=
           new twoFactor(nodes,value)
}