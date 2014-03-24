package gp.classification

import breeze.linalg.{*, diag, DenseMatrix, DenseVector}
import gp.classification.EpParameterEstimator.{AvgBasedStopCriterion, SiteParams}
import utils.KernelRequisites
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
	val (input,targets) = (classInput.trainInput,classInput.targets)
	val kernelMatrix:kernelMatrixType = buildKernelMatrix(kernelFun,input)
	val epParamEstimator:EpParameterEstimator = new EpParameterEstimator(kernelMatrix,targets,stopCriterion)
	epParamEstimator.estimateSiteParams
  }

  /*
  ni_site_param <- site_params$ni_site_param
  tau_site_param <- site_params$tau_site_param
  inputs <- training_set$inputs
  targets <- training_set$targets
  n <- nrow(kernelMatrix)
  ni_diag_vector <- as.numeric(sqrt(tau_site_param))
  lower_triang <- t(chol(diag(n) + outer(ni_diag_vector,ni_diag_vector)*kernelMatrix))
  temp <- forwardsolve(l=lower_triang,x=(ni_diag_vector * kernelMatrix) %*% ni_site_param)
  z_vector <- ni_diag_vector * backsolve(r=t(lower_triang),x=temp)
  test_train_cov_matrix <- covariance_matrix(test_set,inputs)
  f_mean <- test_train_cov_matrix %*% (ni_site_param - z_vector)
  v_vector <- forwardsolve(l=lower_triang,x=ni_diag_vector * t(test_train_cov_matrix))
  f_variance <- covariance_matrix(test_set,test_set) - (t(v_vector) %*% v_vector)
  f_variance1 <- Reduce(function(array,i){array[i] = f_variance[i,i]; array},1:nrow(test_set),array(NA,c(1,nrow(test_set))))
  propability_of_class_1 <- pnorm(t(f_mean)/sqrt(1+f_variance1))
  propability_of_class_1
  
   */
  
  def classify(input:AfterEstimationClassifierInput,kernelMatrix:Option[DenseMatrix[Double]]):classifyOut = {
	val (trainInput,testInput,targets) = (input.trainInput,input.testInput,input.targets)
	val (siteParams,lowerTriangular) = input.learnParams.getOrElse(
	  trainClassifier(ClassifierInput(trainInput = trainInput,targets = targets,initHyperParams = input.hyperParams)))
	val testSetSize = testInput.rows
	val kernelMatrix_ = kernelMatrix.getOrElse(buildKernelMatrix(kernelFun,trainInput))
	val tauDiagVector:DenseVector[Double] = sqrt(siteParams.tauSiteParams)
	val rhs:DenseVector[Double] = (kernelMatrix_(::,*) :* tauDiagVector) * siteParams.niSiteParams
	val tempSol:DenseVector[Double] = forwardSolve(L = lowerTriangular,b = rhs)
	val zVector:DenseVector[Double] = tauDiagVector :* backSolve(R = lowerTriangular.t,b = tempSol)
	val testTrainCovMatrix:kernelMatrixType = buildKernelMatrix(kernelFun,testInput,trainInput)
	assert(testTrainCovMatrix.rows == testInput.rows && testTrainCovMatrix.cols == trainInput.rows)
	val fMean:DenseVector[Double] = testTrainCovMatrix * (siteParams.niSiteParams - zVector)
	assert(fMean.length == testInput.rows)
	val rhs1:DenseMatrix[Double] = (testTrainCovMatrix.t)(::,*) :* tauDiagVector
	val vMatrix:DenseMatrix[Double] = forwardSolve(L = lowerTriangular,b = rhs1)
	val fVariance:DenseMatrix[Double] = buildKernelMatrix(kernelFun,testInput) - (vMatrix.t * vMatrix)
	assert(fVariance.rows == testInput.rows && fVariance.cols == testSetSize)
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

  case class AfterEstimationClassifierInput(trainInput:DenseMatrix[Double],
							 testInput:DenseMatrix[Double],targets:DenseVector[Int],
							 learnParams:Option[learnParams],hyperParams:KernelFuncHyperParams)

  case class ClassifierInput(trainInput:DenseMatrix[Double],targets:DenseVector[Int],initHyperParams:KernelFuncHyperParams)

}
