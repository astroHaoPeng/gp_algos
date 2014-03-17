package gp.datasetstests

import breeze.linalg.{DenseVector, DenseMatrix}
import utils.IOUtilities
import gp.classification.{EpParameterEstimator, MarginalLikelihoodEvaluator, HyperParamsOptimization, GpClassifier}
import utils.KernelRequisites.{GaussianRbfParams, GaussianRbfKernel}
import gp.classification.GpClassifier.{ClassifierInput, AfterEstimationClassifierInput}
import gp.classification.MarginalLikelihoodEvaluator.HyperParameterOptimInput
import gp.classification.HyperParamsOptimization.BreezeLBFGSOptimizer
import gp.classification.EpParameterEstimator.AvgBasedStopCriterion
import breeze.numerics.exp

/**
 * Created by mjamroz on 14/03/14.
 */
class CancerClassificationTest {

  val initRbfParams = GaussianRbfParams(alpha = exp(3.6),gamma = exp(5.1))
//  val initRbfParams = GaussianRbfParams(alpha = 1.,gamma = 1.)
  val eps = 2.5
  val stopCriterion:EpParameterEstimator.stopCriterionFunc = new AvgBasedStopCriterion(eps)
  val rbfKernel = GaussianRbfKernel(rbfParams = initRbfParams)
  val gpClassfier = new GpClassifier(rbfKernel,stopCriterion)

  def loadDataSet:DenseMatrix[Double] = {
	val cancerMatrix:DenseMatrix[Double] = IOUtilities.csvFileToDenseMatrix("cancer.csv")
	cancerMatrix(::,cancerMatrix.cols-1) :=
	  cancerMatrix(::,cancerMatrix.cols-1).mapValues{elem => if (elem == 2.) -1 else 1.}
	cancerMatrix(::,1 until cancerMatrix.cols)
  }

  def test = {
	val wholeDataSet:DenseMatrix[Double] = loadDataSet
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	val learnParams = gpClassfier.trainClassifier(ClassifierInput(trainInput = input,targets = targets,hyperParams = null))
	gpClassfier.classify(AfterEstimationClassifierInput(trainInput = input,testInput = input(5,::),
	  targets = targets,learnParams = Some(learnParams),hyperParams = initRbfParams),None)
  }

  def testWithParamOptimization = {
	val wholeDataSet:DenseMatrix[Double] = loadDataSet
	val input:DenseMatrix[Double] = wholeDataSet(::,0 until (wholeDataSet.cols-1))
	val targets:DenseVector[Int] = wholeDataSet(::,wholeDataSet.cols-1).mapValues(_.toInt)
	val marginalEvaluator = new MarginalLikelihoodEvaluator(stopCriterion,rbfKernel)
	val hyperParamOptimizer = new BreezeLBFGSOptimizer(marginalEvaluator)
	val optimizedParams = hyperParamOptimizer.optimizeHyperParams(
	  ClassifierInput(trainInput = input,targets = targets,hyperParams = rbfKernel.rbfParams))
	val newGpClassifier = new GpClassifier(optimizedParams.asInstanceOf[GaussianRbfKernel],stopCriterion)
	val learnParams = newGpClassifier.trainClassifier(ClassifierInput(trainInput = input,targets = targets,hyperParams = null))
	newGpClassifier.classify(AfterEstimationClassifierInput(trainInput = input,testInput = input(5,::),
		targets = targets,learnParams = Some(learnParams),hyperParams = optimizedParams),None)
  }


}

object CancerClassificationTest{

  def main(args:Array[String]):Unit = {
	println(new CancerClassificationTest().test)
  }

}
