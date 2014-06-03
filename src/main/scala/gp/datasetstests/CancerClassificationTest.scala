package gp.datasetstests

import breeze.linalg.{DenseVector, DenseMatrix}
import utils.IOUtilities
import gp.classification._
import utils.KernelRequisites.GaussianRbfParams
import gp.classification.GpClassifier.ClassifierInput
import gp.classification.HyperParamsOptimization.{ApacheCommonsOptimizer, GradientHyperParamsOptimizer}
import gp.classification.EpParameterEstimator.AvgBasedStopCriterion
import breeze.numerics.exp
import org.slf4j.LoggerFactory
import scala.Some
import gp.classification.GpClassifier.AfterEstimationClassifierInput
import utils.KernelRequisites.GaussianRbfParams
import gp.classification.GpClassifier.ClassifierInput
import utils.KernelRequisites.GaussianRbfKernel
import gp.classification.MeshHyperParamsLogLikelihoodEvaluator.HyperParamsMeshValues
import optimization.Optimization.BreezeLbfgsOptimizer

/**
 * Created by mjamroz on 14/03/14.
 */
class CancerClassificationTest {

  val logger = LoggerFactory.getLogger(CancerClassificationTest.getClass)

  //val initRbfParams = GaussianRbfParams(alpha = exp(4.34),gamma = sqrt(exp(5.1)))
  //val initRbfParams = GaussianRbfParams(alpha = 13.436624999999998,gamma = 164.021)
  //2.046,164.021
  //val initRbfParams = GaussianRbfParams(alpha = 40887.67,gamma = 164.021)
  //val initRbfParams = GaussianRbfParams(alpha = 1.,gamma = 1.)
  //val initRbfParams = GaussianRbfParams(alpha = 0.1,gamma = 0.3)
  val initRbfParams = GaussianRbfParams(signalVar = 3.58,lengthScales = DenseVector(12.),noiseVar = 0.)

  val eps = 0.01
  val stopCriterion:EpParameterEstimator.stopCriterionFunc = new AvgBasedStopCriterion(eps)
  val rbfKernel = GaussianRbfKernel(rbfParams = initRbfParams)
  val gpClassifier = new GpClassifier(rbfKernel,stopCriterion)

  val probabsToClasses:DenseVector[Double] => DenseVector[Int] = {probabs =>
	probabs.mapValues {probab => if (probab < 0.5) -1 else 1}
  }

  val compareTestTargetsWithRightLabels:(DenseVector[Int],DenseVector[Int]) => Int = {
	(predicted,right) =>

	  (0 until predicted.size).foldLeft(0){
		case (errorNum,index) => if (predicted(index) != right(index)){
		  errorNum + 1
		} else {
		  errorNum
		}
	  }
  }

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
	val learnParams = gpClassifier.trainClassifier(ClassifierInput(trainInput = trainSet,targets = targets,initHyperParams = initRbfParams))
	logger.info(s"Marginal log likelihood = ${learnParams._1.marginalLogLikelihood.get}")
	val targetsForTestSet = gpClassifier.classify(AfterEstimationClassifierInput(trainInput = trainSet,testInput = testSet,
	  targets = targets,learnParams = Some(learnParams),hyperParams = initRbfParams),None)
	targetsForTestSet
  }

  def test(limit:Option[Int]) = {
	val wholeDataSet:DenseMatrix[Double] = loadDataSet(limit)
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	testWithTrainAndTestSet(input,targets,input(1,::))
  }

  def testTrainSet(limit:Option[Int]):(DenseVector[Double],DenseVector[Int],Double) = {
	val wholeDataSet:DenseMatrix[Double] = loadDataSet(limit)
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	val probabs = testWithTrainAndTestSet(input,targets,input)
	val predictedClasses = probabsToClasses(probabs)
	val numOfErrors = compareTestTargetsWithRightLabels(predictedClasses,targets)
	(probabs,predictedClasses,numOfErrors.toDouble / targets.length)
  }

  def testSetWithRatio(limit:Option[Int],ratio:Double) = {
	require(ratio > 0.0 && ratio <= 1.0)
	val wholeDataSet:DenseMatrix[Double] = loadDataSet(limit)
	val dsLength:Int = wholeDataSet.rows
	val numOfTrainCases = (ratio * wholeDataSet.rows).toInt
	val input:DenseMatrix[Double] = wholeDataSet(0 until numOfTrainCases,0 until (wholeDataSet.cols-1))
	val tempTargets:DenseVector[Double] = wholeDataSet(::,wholeDataSet.cols-1)
	val targets:DenseVector[Int] = (tempTargets(0 until numOfTrainCases)).mapValues(_.toInt)
	assert(targets.forallValues{label => label == 1 || label == -1})
	val testSet:DenseMatrix[Double] = wholeDataSet(numOfTrainCases until dsLength,0 until (wholeDataSet.cols-1))
	val testTargets:DenseVector[Int] = tempTargets(numOfTrainCases until dsLength).mapValues(_.toInt)
	assert(testTargets.forallValues{label => label == 1 || label == -1})
	val probabs = testWithTrainAndTestSet(input,targets,testSet)
	assert(probabs.length == dsLength - numOfTrainCases)
	val numOfErrors = compareTestTargetsWithRightLabels(probabsToClasses(probabs),testTargets)
	(probabs,testTargets,numOfErrors.toDouble / testTargets.length)
  }

  def testWithParamOptimization(limit:Option[Int]) = {
	val wholeDataSet:DenseMatrix[Double] = loadDataSet(limit)
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	val marginalEvaluator = new MarginalLikelihoodEvaluator(stopCriterion,rbfKernel)
	val hyperParamOptimizer = new GradientHyperParamsOptimizer(marginalEvaluator,new BreezeLbfgsOptimizer)
	//val hyperParamOptimizer = new ApacheCommonsOptimizer(marginalEvaluator)
	val optimizedParams = hyperParamOptimizer.optimizeHyperParams(
	  ClassifierInput(trainInput = input,targets = targets,initHyperParams = rbfKernel.rbfParams))
	val newGpClassifier = new GpClassifier(rbfKernel.changeHyperParams(optimizedParams.toDenseVector),stopCriterion)
	val learnParams = newGpClassifier.trainClassifier(ClassifierInput(trainInput = input,targets = targets,initHyperParams = optimizedParams))
	newGpClassifier.classify(AfterEstimationClassifierInput(trainInput = input,testInput = input(35,::),
		targets = targets,learnParams = Some(learnParams),hyperParams = optimizedParams),None)
  }

  def evaluateHyperParamsMesh(limit:Option[Int]):HyperParamsMeshValues = {
	val wholeDataSet = loadDataSet(limit)
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	val marginalEvaluator = new MarginalLikelihoodEvaluator(stopCriterion,rbfKernel)
	val meshEvaluator = new MeshHyperParamsLogLikelihoodEvaluator(marginalEvaluator)
	val (alphaRange,gammaRange) = (Range.Double(0.5,100.,8.),Range.Double(0.01,10.,1.))
	meshEvaluator.evaluate(IndexedSeq(alphaRange,gammaRange),
	  ClassifierInput(trainInput = input,targets = targets,initHyperParams = rbfKernel.rbfParams))
  }

}

object CancerClassificationTest{

  def main(args:Array[String]):Unit = {
	//println(new CancerClassificationTest().testSetWithRatio(None,0.7))
	val meshValues = new CancerClassificationTest().evaluateHyperParamsMesh(Some(200))
	meshValues.writeToFile("200_3.dat")
	println(meshValues)
  }

}
