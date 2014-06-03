package gp.datasetstests

import breeze.numerics._
import utils.KernelRequisites.GaussianRbfParams
import utils.KernelRequisites.GaussianRbfKernel
import org.slf4j.LoggerFactory
import gp.regression.GpPredictor
import utils.StatsUtils.{NormalDistributionSampler, GaussianDistribution}
import breeze.linalg.DenseVector
import gp.regression.GpPredictor.PredictionInput

/**
 * Created by mjamroz on 11/04/14.
 */
class BostonPredictionDataSetTest {

  import utils.DataSetsTestingUtils._

  val logger = LoggerFactory.getLogger(classOf[BostonPredictionDataSetTest])

  type valueExtractingFunc = GaussianDistribution => DenseVector[Double]

  //val (alpha,gamma,beta) = (1.,-1.,sqrt(0.0125))
  val (alpha,gamma,beta) = (exp(6.1),sqrt(exp(5.)),sqrt(0.0015))
  val defaultRbfParams:GaussianRbfParams = GaussianRbfParams(signalVar = alpha,
	lengthScales = DenseVector(gamma),noiseVar = beta)
  val gaussianKernel = GaussianRbfKernel(defaultRbfParams)
  val gpPredictor = new GpPredictor(gaussianKernel)

  val gaussianMeanExtractingFunc:valueExtractingFunc = _.mean
  val samplingExtractingFunc:valueExtractingFunc = {
	normalDistr => new NormalDistributionSampler(normalDistr).sample
  }

  private def squaredError(vec1:DenseVector[Double],vec2:DenseVector[Double]):Double = {
	require(vec1.length == vec2.length)
	val diffs:DenseVector[Double] = vec1 - vec2
	(0 until vec1.length).foldLeft(0.){
	  case (acc:Double,index:Int) =>
	  	acc + diffs(index)*diffs(index)
	}
  }

  def evaluatePredictor(ratio:Option[Double],optimizeParams:Boolean,extractingFunc:valueExtractingFunc) = {
	loadDataSet(DataLoadSpec(fileName = "boston.csv",divRatio = ratio)) match {
	  case Right(dataSet) =>
		val predictionInput = PredictionInput(trainingData = dataSet.trainingData,
		  testData = dataSet.testData,sigmaNoise = None,targets = dataSet.trainingLabels)
		val (gaussianDistr,logLikelihood) = if (!optimizeParams){
		  gpPredictor.predict(predictionInput)
		} else {
		  val (distr,ll,_) = gpPredictor.predictWithParamsOptimization(predictionInput,false)
		  (distr,ll)
		}
		logger.info(s"Log likelihood = $logLikelihood")
		val predictedValues = extractingFunc(gaussianDistr)
		val error = squaredError(predictedValues,dataSet.testLabels)
		logger.info(s"Squared Error = ${error}")

	  case Left(ex) => throw ex
	}

  }



}

object BostonPredictionDataSetTest {

  def main(args:Array[String]) = {
	val test = new BostonPredictionDataSetTest
	test.evaluatePredictor(Some(0.7),false,extractingFunc = test.gaussianMeanExtractingFunc)
  }

}
