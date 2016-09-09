package sparknlp.optimization

import breeze.linalg.{SparseVector, Vector}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer

/**
 * Created by Mr.gong on 2016/9/8.
 */
object crfsGradientDesentWithHash extends Logging {

    def runMiniBatchSGD(
                         data:RDD[Seq[(Char,Int)]],
                         numLabels:Int,
                         unigram:Set[Seq[Int]],
                         gradient: CrfsGradient,
                         updater: Updater,
                         stepSize: Double,
                         numIterations: Int,
                         regParam: Double,
                         miniBatchFraction: Double,
                         numFeatures:Int): (Vector[Double], Array[Double]) = {


      val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
      // Record previous weight and current one to calculate solution vector difference

      var previousWeights: Option[Vector[Double]] = None
      var currentWeights: Option[Vector[Double]] = None

      val numExamples = data.count()

      if (numExamples * miniBatchFraction < 1) {
        logWarning("The miniBatchFraction is too small")
      }

      // Initialize weights as a column vector
      val initialWeights=SparseVector.zeros[Double](numFeatures)
      var weights:Vector[Double]= initialWeights
      //val n = weights.size

      // if no data, return initial weights to avoid NaNs
      if (numExamples == 0) {
        logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
        return (initialWeights, stochasticLossHistory.toArray)
      }

      /**
       * For the first iteration, the regVal will be initialized as sum of weight squares
       * if it's L2 updater; for L1 updater, the same logic is followed.
       */
      var regVal = updater.compute(
        weights, SparseVector.zeros(weights.size), 0, 1, regParam)._2

      var i = 1


      while (i <= numIterations) {
        val bcWeights = data.context.broadcast(weights)
        // Sample a subset (fraction miniBatchFraction) of the total data
        // compute and sum up the subgradients on this subset (this is one map-reduce)
        val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + i)
          .treeAggregate((SparseVector.zeros[Double](numFeatures), 0.0, 0L))(
            seqOp = (c, v) => {
              // c: (grad, loss, count), v: (label, features)
              val l= gradient.compute(v,bcWeights.value,numLabels,unigram,c._1)
              (c._1, c._2 + l, c._3 + 1)
            },
            combOp = (c1, c2) => {
              // c: (grad, loss, count)
              (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
            })

        if (miniBatchSize > 0) {
          /**
           * lossSum is computed using the weights from the previous iteration
           * and regVal is the regularization value computed in the previous iteration as well.
           */
          stochasticLossHistory.append(lossSum / miniBatchSize + regVal)
          val update = updater.compute(
            weights,gradientSum / miniBatchSize.toDouble,
            stepSize, i, regParam)
          weights = update._1
          regVal = update._2

          previousWeights = currentWeights
          currentWeights = Some(weights)
        } else {
          logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
        }
        i += 1
      }

      logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
        stochasticLossHistory.takeRight(10).mkString(", ")))

      (weights, stochasticLossHistory.toArray)

    }

    /**
     * Alias of [[runMiniBatchSGD]] with convergenceTol set to default value of 0.001.
     */
    def runMiniBatchSGD(
                         data: RDD[Seq[(Char,Int)]],
                         numLabels:Int,
                         unigram:Set[Seq[Int]],
                         gradient: CrfsGradient,
                         updater: Updater,
                         stepSize: Double,
                         numIterations: Int,
                         regParam: Double,
                         miniBatchFraction: Double): (Vector[Double], Array[Double],Map[String,Int]) =
      crfsGradientDesentWithHash.runMiniBatchSGD(data,numLabels,unigram,gradient, updater, stepSize, numIterations,
        regParam, miniBatchFraction)

  }

