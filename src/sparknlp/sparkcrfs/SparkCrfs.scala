package sparknlp.sparkcrfs

/**
 * Created by Mr.gong on 2016/8/15.
 */
import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import sparknlp.features.{GenerateHashFeatures, generateAllFeatures, crfsFeature}
import sparknlp.inference.LinearCliqueTree
import sparknlp.optimization.{crfsGradientDescent, SimpleUpdater, CrfsGradient}
import breeze.linalg.Vector

class SparkCrfsModel(weights: Vector[Double],bcConditionalFeatureMap:Broadcast[Map[String,Int]]) extends Serializable{

  def predict(testData: RDD[Seq[(Char,Int)]],numLabels:Int,unigram:Set[Seq[Int]]) = {
    val bcWeights = testData.context.broadcast(weights)
    testData.mapPartitions { iter =>
      val w = bcWeights.value
      val cfm=bcConditionalFeatureMap.value
      iter.map(v => predictPoint(v,numLabels,unigram,w,cfm))
    }
  }

  def predictPoint(data:Seq[(Char,Int)],numLabels:Int,unigram:Set[Seq[Int]],weights: Vector[Double],conditionalFeatureMap:Map[String,Int])={
    val numFeatures=weights.length
    val allFeatures:Seq[crfsFeature]=generateAllFeatures.run(data,numLabels,unigram,conditionalFeatureMap)
    val cliqueTree=new LinearCliqueTree(data.length,numLabels)
    cliqueTree.addFeatures(allFeatures,weights)
    val calibratedTree=cliqueTree.maxSumBP()
    val labels=calibratedTree.computeMaxMarginalsBP()
    data zip labels
  }

 // protected def accurcy
}


class SparkCrfsHashModel(weights: Vector[Double]) extends Serializable{

  def predict(testData: RDD[Seq[(Char,Int)]],numLabels:Int,unigram:Set[Seq[Int]]) = {
    val bcWeights = testData.context.broadcast(weights)
    testData.mapPartitions { iter =>
      val w = bcWeights.value
      iter.map(v => predictPoint(v,numLabels,unigram,w))
    }
  }

  def predictPoint(data:Seq[(Char,Int)],numLabels:Int,unigram:Set[Seq[Int]],weights: Vector[Double])={
    val numFeatures=weights.length
    val allFeatures:Seq[crfsFeature]=GenerateHashFeatures.run(data,numLabels,unigram,numFeatures)
    val cliqueTree=new LinearCliqueTree(data.length,numLabels)
    cliqueTree.addFeatures(allFeatures,weights)
    val calibratedTree=cliqueTree.maxSumBP()
    val labels=calibratedTree.computeMaxMarginalsBP()
    data zip labels
  }

  // protected def accurcy
}


class SparkCrfsModelWithSGD (
      private var stepSize: Double,
      private var numIterations: Int,
      private var numLables:Int,
      private var unigram:Set[Seq[Int]],
      private var miniBatchFraction: Double) extends Logging
{
  private val gradient=new CrfsGradient
  private val updater= new SimpleUpdater()
  val optimizer= new crfsGradientDescent(gradient,updater,unigram)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setMiniBatchFraction(miniBatchFraction)
    .setNumLabers(numLables)


  //def this() = this(1.0, 100, 1.0)

  def createModel(weights: Vector[Double],bcConditionalFeaturesMap:Broadcast[Map[String,Int]])= new SparkCrfsModel(weights,bcConditionalFeaturesMap)

  def run(input:  RDD[Seq[(Char,Int)]]):SparkCrfsModel = {

    if (input.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Check the data properties before running the optimizer
 //   if (validateData && !validators.forall(func => func(input))) {
  //    throw new SparkException("Input validation failed.")
   // }

    val (weights,bcConditionalFeaturesMap) = optimizer.optimize(input)

    // Warn at the end of the run as well, for increased visibility.
    if (input.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Unpersist cached data
 //   if (.getStorageLevel != StorageLevel.NONE) {
 //     data.unpersist(false)
 //   }

    createModel(weights,bcConditionalFeaturesMap)
  }
}


object SparkCrfsWithSGD{
  def train(
             input:RDD[Seq[(Char,Int)]],
             numIterations: Int,
             numLables:Int,
             stepSize: Double,
             miniBatchFraction: Double,
             unigram:Set[Seq[Int]]): SparkCrfsModel = {
    new SparkCrfsModelWithSGD(stepSize, numIterations,numLables,unigram,miniBatchFraction)
      .run(input)
  }
}