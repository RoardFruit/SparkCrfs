package sparknlp.optimization

/**
 * Created by Mr.gong on 2016/9/1.
 */
import org.apache.spark.broadcast.Broadcast
import sparknlp.features.{crfsFeature, generateAllFeatures, GenerateConditonalFeaturesMap}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}

import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.rdd.RDD


@DeveloperApi
object LBFGSforCRFs2 extends Logging {
  /**
   * Run Limited-memory BFGS (L-BFGS) in parallel.
   * Averaging the subgradients over different partitions is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data - Input data for L-BFGS. RDD of the set of data examples, each of
   *               the form (label, [feature values]).
   * @param gradient - Gradient object (used to compute the gradient of the loss function of
   *                   one single data example)
   * @param updater - Updater function to actually perform a gradient step in a given direction.
   * @param numCorrections - The number of corrections used in the L-BFGS update.
   * @param convergenceTol - The convergence tolerance of iterations for L-BFGS which is must be
   *                         nonnegative. Lower values are less tolerant and therefore generally
   *                         cause more iterations to be run.
   * @param maxNumIterations - Maximal number of iterations that L-BFGS can be run.
   * @param regParam - Regularization parameter
   *
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the loss
   *         computed for every iteration.
   */
  def runLBFGS(
                data: RDD[Seq[(Char,Int)]],
                numLabels:Int,
                unigram:Set[Seq[Int]],
                gradient: CrfsGradient,
                updater: Updater,
                numCorrections: Int,
                convergenceTol: Double,
                maxNumIterations: Int,
                regParam: Double): (Vector[Double], Array[Double],Map[String,Int]) = {

    val lossHistory = mutable.ArrayBuilder.make[Double]

    val numExamples = data.count()

    val conditionalFeatureMap=GenerateConditonalFeaturesMap.run(data,numLabels,unigram,1)
    //val bcConditionalFeatureMap:Broadcast[Map[String,Int]]= data.context.broadcast(conditionalFeatureMap)
    val n=conditionalFeatureMap.size+numLabels*numLabels

    val transformedRDD=for{
      sentence<-data
    } yield {
        val crfsFeatures=generateAllFeatures.run(sentence,numLabels,unigram,conditionalFeatureMap)
        val featureCounts=SparseVector.zeros[Double](n)
        for{
          crfsFeature(nodes,assignments,idx)<-crfsFeatures
          if sentence(nodes)._2==assignments
        } featureCounts(idx)+=1

        def linearIndex(row: Int, col: Int)=row-1 + numLabels * (col-1)
        (sentence.indices.map(x=>sentence(x)._2) zip sentence.indices.tail.map(x=>sentence(x)._2)) foreach (x=>featureCounts(linearIndex(x._1,x._2))+=1)
        (featureCounts,crfsFeatures,sentence.length)
      }

    transformedRDD.cache()


    // Initialize weights as a column vector
    val initialWeights=DenseVector.zeros[Double](n)
    //var weights:Vector[Double]= initialWeights

    val costFun =
      new CostFun(transformedRDD, numLabels,gradient, updater, regParam, numExamples)

    val lbfgs = new LBFGS[DenseVector[Double]](maxNumIterations, numCorrections, convergenceTol)

    val states =
      lbfgs.iterations(new CachedDiffFunction(costFun), initialWeights)

    /**
     * NOTE: lossSum and loss is computed using the weights from the previous iteration
     * and regVal is the regularization value computed in the previous iteration as well.
     */
    var state = states.next()
    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }
    lossHistory += state.value
    val weights = state.x

    val lossHistoryArray = lossHistory.result()

    logInfo("LBFGS.runLBFGS finished. Last 10 losses %s".format(
      lossHistoryArray.takeRight(10).mkString(", ")))

    (weights, lossHistoryArray,conditionalFeatureMap)
  }

  /**
   * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
   * at a particular point (weights). It's used in Breeze's convex optimization routines.
   */
  private class CostFun(
                         data:RDD[(SparseVector[Double],Seq[crfsFeature],Int)],
                         numLabels:Int,
                         gradient: CrfsGradient,
                         updater: Updater,
                         regParam: Double,
                         numExamples: Long) extends DiffFunction[DenseVector[Double]] with Serializable{

    override def calculate(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {
      // Have a local copy to avoid the serialization of CostFun object which is not serializable.
      // val w = weights
      val n = weights.size
      val bcW = data.context.broadcast(weights)
      //val localGradient = gradient

      val (gradientSum, lossSum) = data.treeAggregate((DenseVector.zeros[Double](n), 0.0))(
        seqOp = (c, v) => (c, v) match { case ((grad, loss), v) =>
          val l = gradient.compute(
            v,bcW.value,numLabels,grad)
          (grad, loss + l)
        },
        combOp = (c1, c2) => (c1, c2) match { case ((grad1, loss1), (grad2, loss2)) =>
          axpy(1.0, grad2, grad1)
          (grad1, loss1 + loss2)
        })

      /**
       * regVal is sum of weight squares if it's L2 updater;
       * for other updater, the same logic is followed.
       */
      val regVal = updater.compute(weights, DenseVector.zeros(n), 0, 1, regParam)._2

      val loss = lossSum / numExamples + regVal
      /**
       * It will return the gradient part of regularization using updater.
       *
       * Given the input parameters, the updater basically does the following,
       *
       * w' = w - thisIterStepSize * (gradient + regGradient(w))
       * Note that regGradient is function of w
       *
       * If we set gradient = 0, thisIterStepSize = 1, then
       *
       * regGradient(w) = w - w'
       *
       * TODO: We need to clean it up by separating the logic of regularization out
       *       from updater to regularizer.
       */
      // The following gradientTotal is actually the regularization part of gradient.
      // Will add the gradientSum computed from the data with weights in the next step.
      val gradientTotal = weights.copy
      axpy(-1.0, updater.compute(weights, DenseVector.zeros(n), 1, 1, regParam)._1.toDenseVector, gradientTotal)

      // gradientTotal = gradientSum / numExamples + gradientTotal
      axpy(1.0 / numExamples, gradientSum, gradientTotal)

      (loss, gradientTotal)
    }
  }
}