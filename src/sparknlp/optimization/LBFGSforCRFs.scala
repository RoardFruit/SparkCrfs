package sparknlp.optimization

/**
 * Created by Mr.gong on 2016/8/30.
 */


import org.apache.spark.broadcast.Broadcast
import sparknlp.features.GenerateConditonalFeaturesMap

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{SparseVector, DenseVector, Vector,axpy}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}

import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.rdd.RDD

/**
 * :: DeveloperApi ::
 * Class used to solve an optimization problem using Limited-memory BFGS.
 * Reference: [[http://en.wikipedia.org/wiki/Limited-memory_BFGS]]
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
@DeveloperApi
class LBFGSforCRFs(private var gradient: CrfsGradient, private var updater: Updater,private var unigram:Set[Seq[Int]])
  extends Serializable with Logging {

  private var numCorrections = 10
  private var convergenceTol = 1E-4
  private var maxNumIterations = 100
  private var regParam = 0.0
  private var numLabers=1
  private var minCount=1

  /**
   * Set the number of corrections used in the LBFGS update. Default 10.
   * Values of numCorrections less than 3 are not recommended; large values
   * of numCorrections will result in excessive computing time.
   * 3 < numCorrections < 10 is recommended.
   * Restriction: numCorrections > 0
   */
  def setNumCorrections(corrections: Int): this.type = {
    assert(corrections > 0)
    this.numCorrections = corrections
    this
  }

  /**
   * Set the convergence tolerance of iterations for L-BFGS. Default 1E-4.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * This value must be nonnegative. Lower convergence values are less tolerant
   * and therefore generally cause more iterations to be run.
   */
  def setConvergenceTol(tolerance: Double): this.type = {
    this.convergenceTol = tolerance
    this
  }

  /**
   * Set the maximal number of iterations for L-BFGS. Default 100.
   * @deprecated use [[LBFGS#setNumIterations]] instead
   */
  @deprecated("use setNumIterations instead", "1.1.0")
  def setMaxNumIterations(iters: Int): this.type = {
    this.setNumIterations(iters)
  }

  /**
   * Set the maximal number of iterations for L-BFGS. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.maxNumIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for L-BFGS.
   */
  def setGradient(gradient: CrfsGradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  def setNumLabers(numLabers:Int):this.type = {
    this.numLabers = numLabers
    this
  }

  def setMinCount(minCount:Int):this.type = {
    this.minCount = minCount
    this
  }

  def setUnigram(unigram:Set[Seq[Int]]):this.type = {
    this.unigram = unigram
    this
  }

   def optimize(data: RDD[Seq[(Char, Int)]]): Vector[Double]= {
    val (weights, _,_) = LBFGSforCRFs.runLBFGS(
      data,
      numLabers,
      unigram,
      gradient,
      updater,
      numCorrections,
      convergenceTol,
      maxNumIterations,
      minCount,
      regParam)
    weights
  }

}

/**
 * :: DeveloperApi ::
 * Top-level method to run L-BFGS.
 */
@DeveloperApi
object LBFGSforCRFs extends Logging {
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
                minCount:Int,
                regParam: Double): (Vector[Double], Array[Double],Broadcast[Map[String,Int]]) = {

    val lossHistory = mutable.ArrayBuilder.make[Double]

    val numExamples = data.count()

    val conditionalFeatureMap=GenerateConditonalFeaturesMap.run(data,numLabels,unigram,minCount)
    val bcConditionalFeatureMap:Broadcast[Map[String,Int]]= data.context.broadcast(conditionalFeatureMap)

    val n=conditionalFeatureMap.size+numLabels*numLabels
    // Initialize weights as a column vector
    val initialWeights=DenseVector.zeros[Double](n)
    //var weights:Vector[Double]= initialWeights

    val costFun =
      new CostFun(data, numLabels,unigram,bcConditionalFeatureMap,gradient, updater, regParam, numExamples)

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

    (weights, lossHistoryArray,bcConditionalFeatureMap)
  }

  /**
   * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
   * at a particular point (weights). It's used in Breeze's convex optimization routines.
   */
  private class CostFun(
                         data: RDD[Seq[(Char, Int)]],
                         numLabels:Int,
                         unigram:Set[Seq[Int]],
                         bcConditionalFeatureMap:Broadcast[Map[String,Int]],
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
            v,bcW.value,numLabels,unigram,bcConditionalFeatureMap.value,grad)
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