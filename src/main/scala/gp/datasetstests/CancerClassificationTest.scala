package gp.datasetstests

import breeze.linalg.{DenseVector, DenseMatrix}
import utils.IOUtilities
import gp.classification.{EpParameterEstimator, MarginalLikelihoodEvaluator, HyperParamsOptimization, GpClassifier}
import utils.KernelRequisites.{GaussianRbfParams, GaussianRbfKernel}
import gp.classification.GpClassifier.{ClassifierInput, AfterEstimationClassifierInput}
import gp.classification.MarginalLikelihoodEvaluator.HyperParameterOptimInput
import gp.classification.HyperParamsOptimization.{ApacheCommonsOptimizer, BreezeLBFGSOptimizer}
import gp.classification.EpParameterEstimator.AvgBasedStopCriterion
import breeze.numerics.{sqrt, exp}
import org.slf4j.LoggerFactory

/**
 * Created by mjamroz on 14/03/14.
 */
class CancerClassificationTest {

  val logger = LoggerFactory.getLogger(CancerClassificationTest.getClass)

  //val initRbfParams = GaussianRbfParams(alpha = exp(4.34),gamma = sqrt(exp(5.1)))
  //val initRbfParams = GaussianRbfParams(alpha = 13.436624999999998,gamma = 164.021)
  //2.046,164.021
  //val initRbfParams = GaussianRbfParams(alpha = 40887.67,gamma = 164.021)
  val initRbfParams = GaussianRbfParams(alpha = 1.,gamma = 1.)
  //val initRbfParams = GaussianRbfParams(alpha = 0.1,gamma = 0.3)

  val eps = 0.01
  val stopCriterion:EpParameterEstimator.stopCriterionFunc = new AvgBasedStopCriterion(eps)
  val rbfKernel = GaussianRbfKernel(rbfParams = initRbfParams)
  val gpClassfier = new GpClassifier(rbfKernel,stopCriterion)

  def loadDataSet(limit:Option[Int] = None):DenseMatrix[Double] = {
	val cancerMatrix:DenseMatrix[Double] = IOUtilities.csvFileToDenseMatrix("cancer.csv")
	cancerMatrix(::,cancerMatrix.cols-1) :=
	  cancerMatrix(::,cancerMatrix.cols-1).mapValues{elem => if (elem == 2.) -1. else 1.}
	limit match {
	  case Some(limit_) => cancerMatrix(0 until limit_,1 until cancerMatrix.cols)
	  case None => 	cancerMatrix(::,1 until cancerMatrix.cols)
	}
  }

  def testWithTrainAndTestSet(trainSet:DenseMatrix[Double],targets:DenseVector[Int],
							  testSet:DenseMatrix[Double]):DenseVector[Double] = {
	val learnParams = gpClassfier.trainClassifier(ClassifierInput(trainInput = trainSet,targets = targets,hyperParams = null))
	logger.info(s"Marginal log likelihood = ${learnParams._1.marginalLogLikelihood.get}")
	val targetsForTestSet = gpClassfier.classify(AfterEstimationClassifierInput(trainInput = trainSet,testInput = testSet,
	  targets = targets,learnParams = Some(learnParams),hyperParams = initRbfParams),None)
	targetsForTestSet
  }

  def test(limit:Option[Int]) = {
	val wholeDataSet:DenseMatrix[Double] = loadDataSet(limit)
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	testWithTrainAndTestSet(input,targets,input(35,::))
  }

  def testTrainSet(limit:Option[Int]):(DenseVector[Int],Double) = {
	val wholeDataSet:DenseMatrix[Double] = loadDataSet(limit)
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	val probabs = testWithTrainAndTestSet(input,targets,input)
	val predictedClasses = probabs.mapValues {probab => if (probab < 0.5) -1 else 1}
	val numOfErrors = (0 until predictedClasses.size).foldLeft(0){
	  case (errorNum,index) => if (predictedClasses(index) != targets(index)){
		errorNum + 1
	  } else {
		errorNum
	  }
	}
	(predictedClasses,numOfErrors.toDouble / targets.length)
  }

  def testWithParamOptimization(limit:Option[Int]) = {
	val wholeDataSet:DenseMatrix[Double] = loadDataSet(limit)
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	val marginalEvaluator = new MarginalLikelihoodEvaluator(stopCriterion,rbfKernel)
	val hyperParamOptimizer = new BreezeLBFGSOptimizer(marginalEvaluator)
	//val hyperParamOptimizer = new ApacheCommonsOptimizer(marginalEvaluator)
	val optimizedParams = hyperParamOptimizer.optimizeHyperParams(
	  ClassifierInput(trainInput = input,targets = targets,hyperParams = rbfKernel.rbfParams))
	val newGpClassifier = new GpClassifier(rbfKernel.changeHyperParams(optimizedParams.toDenseVector),stopCriterion)
	val learnParams = newGpClassifier.trainClassifier(ClassifierInput(trainInput = input,targets = targets,hyperParams = null))
	newGpClassifier.classify(AfterEstimationClassifierInput(trainInput = input,testInput = input(5,::),
		targets = targets,learnParams = Some(learnParams),hyperParams = optimizedParams),None)
  }


}

object CancerClassificationTest{

  def main(args:Array[String]):Unit = {
	println(new CancerClassificationTest().test(Some(200)))
  }

}
