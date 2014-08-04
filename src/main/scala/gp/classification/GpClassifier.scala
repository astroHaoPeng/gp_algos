package gp.classification

import breeze.linalg.{*, DenseMatrix, DenseVector}
import gp.classification.EpParameterEstimator.SiteParams
import utils.KernelRequisites.{KernelFuncHyperParams, kernelMatrixType, KernelFunc}
import breeze.numerics.sqrt

/**
 * Created by mjamroz on 13/03/14.
 */
class GpClassifier(kernelFun:KernelFunc,stopCriterion:EpParameterEstimator.stopCriterionFunc) {

  import GpClassifier._
  import utils.MatrixUtils._
  import utils.StatsUtils._

  /*targets must contain values from set {-1,1} */
  def trainClassifier(classInput:ClassifierInput):learnParams = {
	val (kernelMatrix,targets) = (classInput.trainKernelMatrix,classInput.targets)
	val epParamEstimator:EpParameterEstimator = new EpParameterEstimator(kernelMatrix,targets,stopCriterion)
	epParamEstimator.estimateSiteParams
  }
  
  def classify(input:AfterEstimationClassifierInput):classifyOut = {
	val targets = input.targets
	val (siteParams,lowerTriangular) = input.learnParams.getOrElse(
	  trainClassifier(ClassifierInput(trainKernelMatrix = input.trainKernelMatrix,targets = targets,
		initHyperParams = input.hyperParams,trainData = None)))
	val testTrainCovMatrix:kernelMatrixType = input.testTrainKernelMatrix
	val testSetSize = testTrainCovMatrix.rows
	val kernelMatrix_ = input.trainKernelMatrix
	val testKernelMatrix = input.testKernelMatrix
	val tauDiagVector:DenseVector[Double] = sqrt(siteParams.tauSiteParams)
	val rhs:DenseVector[Double] = (kernelMatrix_(::,*) :* tauDiagVector) * siteParams.niSiteParams
	val tempSol:DenseVector[Double] = forwardSolve(L = lowerTriangular,b = rhs)
	val zVector:DenseVector[Double] = tauDiagVector :* backSolve(R = lowerTriangular.t,b = tempSol)
	val fMean:DenseVector[Double] = testTrainCovMatrix * (siteParams.niSiteParams - zVector)
	assert(fMean.length == testSetSize)
	val rhs1:DenseMatrix[Double] = (testTrainCovMatrix.t)(::,*) :* tauDiagVector
	val vMatrix:DenseMatrix[Double] = forwardSolve(L = lowerTriangular,b = rhs1)
	val fVariance:DenseMatrix[Double] = testKernelMatrix - (vMatrix.t * vMatrix)
	assert(fVariance.rows == testSetSize && fVariance.cols == testSetSize)
	val probabilitiesOfClass1 = (0 until testSetSize).foldLeft(DenseVector.zeros[Double](testSetSize)){
	  case (probs,indx) => probs.update(indx,pnorm(fMean(indx)/sqrt(1+fVariance(indx,indx)))); probs
	}
	probabilitiesOfClass1
  }

}

object GpClassifier {

  type classifyOut = DenseVector[Double]

  /*second elem from tuple is an output from cholesky decomposition*/
  type learnParams = (SiteParams,DenseMatrix[Double])

  case class AfterEstimationClassifierInput(targets:DenseVector[Int],
							 learnParams:Option[learnParams],hyperParams:KernelFuncHyperParams,
							 trainKernelMatrix:DenseMatrix[Double],testTrainKernelMatrix:DenseMatrix[Double],
							 testKernelMatrix:DenseMatrix[Double])

  case class ClassifierInput(trainKernelMatrix:DenseMatrix[Double],
							 targets:DenseVector[Int],
							 initHyperParams:KernelFuncHyperParams,trainData:Option[DenseMatrix[Double]])

}
