package gp.tasks

import utils.IOUtilities
import java.io.File
import org.springframework.context.support.GenericXmlApplicationContext
import gp.regression.GpPredictor
import gp.regression.GpPredictor.PredictionInput
import utils.StatsUtils.GaussianDistribution
import breeze.linalg.{DenseMatrix, DenseVector}
import dynamicalsystems.tests.SsmTests

/**
 * Created by mjamroz on 13/07/14.
 */
object MasterThesisRelatedTasks {

  import gp.regression.Co2Prediction._
  import utils.DataSetsTestingUtils._

  val genericContext = new GenericXmlApplicationContext()
  genericContext.load("classpath:config/spring-context.xml")
  genericContext.refresh()

  def writeCo2DataSetToFile = {
	val (co2Train,_) = co2DataToYearWithValue(loadInput,1.0)
	val co2DsFile = new File("src/main/resources/co2/maunaLoa2D.txt")
	IOUtilities.writeVectorsToFile(co2DsFile,co2Train(::,0),co2Train(::,1))
  }

  def gpPosteriorToVecOutput(posterior:GaussianDistribution,wholeDataSet:DenseMatrix[Double]):DenseMatrix[Double] = {
	assert(posterior.dim == wholeDataSet.rows)
	val resultMatrix = DenseMatrix.tabulate[Double](posterior.dim,3){
	  case (row,col) =>
	  	col match {
		  case 0 => wholeDataSet(row,0)
		  case 1 => posterior.mean(row)
		  case 2 => math.sqrt(posterior.sigma(row,row))
		}
	}
	resultMatrix
  }

  def gpPosteriorToVecOutputForBoston(posterior:GaussianDistribution,
									  wholeDataSet:DenseMatrix[Double],targetValues:DenseVector[Double]):DenseMatrix[Double] = {
	assert(posterior.dim == wholeDataSet.rows)
	assert(wholeDataSet.rows == targetValues.length)
	val resultMatrix = DenseMatrix.tabulate[Double](posterior.dim,4){
	  case (row,col) =>
	  	col match {
		  case 0 => row
		  case 1 => posterior.mean(row)
		  case 2 => math.sqrt(posterior.sigma(row,row))
		  case 3 => targetValues(row)
		}
	}
	resultMatrix
  }

  def evaluateGpPredictionOnCo2Ds = {
	val trainTestRatio = 0.7
	val (co2Train,co2Test) = co2DataToYearWithValue(loadInput,trainTestRatio)
	val wholeDataSet = DenseMatrix.vertcat[Double](co2Train,co2Test)
	val co2GpPredictor = genericContext.getBean("co2GpPredictor", classOf[GpPredictor])
	val predInput = PredictionInput(trainingData = co2Train(::, 0).toDenseMatrix.t, sigmaNoise = None,
	  targets = co2Train(::, 1), testData = wholeDataSet(::,0).toDenseMatrix.t)
	val (co2PredictionResult,ll,_) = co2GpPredictor.predictWithParamsOptimization(predInput,true)
	println(s"Log likelihood = $ll, optimal params = ")
	val co2PredFile = new File("src/main/resources/co2/co2PredResults.txt")
	val meanWithStddev = gpPosteriorToVecOutput(co2PredictionResult,wholeDataSet)
	IOUtilities.writeVectorsToFile(co2PredFile,meanWithStddev(::,0),meanWithStddev(::,1),meanWithStddev(::,2))
  }

  def evaluateGpPredictionOnBostonDs = {
	val ratio = 0.7
	val gpPredictor = genericContext.getBean("gpBostonOptimalPredictor",classOf[GpPredictor])
	loadDataSet(DataLoadSpec(fileName = "boston.csv",divRatio = Some(ratio))) match {
	  case Right(dataSet) =>
		val predictionInput = PredictionInput(trainingData = dataSet.trainingData,
		  testData = dataSet.wholeDataSet,sigmaNoise = None,targets = dataSet.trainingLabels)
		val (bostonPosterior,ll,optimalHyperParams) = gpPredictor.predictWithParamsOptimization(predictionInput,true)
		println(s"Log likelihood = $ll")
		println(s"Optimal Hyperparams = ${optimalHyperParams}")
		val bostonFile = new File("src/main/resources/boston/bostonPredResults.txt")
		val bostonVecs = gpPosteriorToVecOutputForBoston(bostonPosterior,dataSet.wholeDataSet,
		  DenseVector.vertcat[Double](dataSet.trainingLabels,dataSet.testLabels))
		IOUtilities.writeVectorsToFile(bostonFile,bostonVecs(::,0),bostonVecs(::,1),bostonVecs(::,2),bostonVecs(::,3))

	  case Left(ex) => throw ex
	}
  }

  def evaluateSsmInference = {
	SsmTests.main(Array.empty[String])
  }

  def main(args:Array[String]):Unit = {
	//writeCo2DataSetToFile
	//evaluateGpPredictionOnCo2Ds
	//evaluateGpPredictionOnBostonDs
	evaluateSsmInference
  }

}
