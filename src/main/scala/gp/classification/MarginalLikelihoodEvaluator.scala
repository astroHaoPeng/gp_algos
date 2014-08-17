package gp.classification

import gp.classification.EpParameterEstimator.SiteParams
import breeze.linalg.{trace, diag, DenseVector, DenseMatrix}
import breeze.numerics.sqrt
import breeze.optimize.DiffFunction
import utils.KernelRequisites._
import utils.MatrixUtils

/**
 * Created by mjamroz on 15/03/14.
 */
class MarginalLikelihoodEvaluator(stopCriterion:EpParameterEstimator.stopCriterionFunc,kernelFunc:KernelFunc) {

  import utils.MatrixUtils._
  import MarginalLikelihoodEvaluator._

  def logLikelihoodWithKernelMatrixPassed(kernelMatrix:DenseMatrix[Double],targets:DenseVector[Int]):Double = {
	val epParameterEstimator = new EpParameterEstimator(kernelMatrix,targets,stopCriterion)
	val (siteParams,_) = epParameterEstimator.estimateSiteParams
	siteParams.marginalLogLikelihood.get
  }

  def logLikelihoodWithoutGrad(trainInput:DenseMatrix[Double],
					targets:DenseVector[Int],hyperParams:DenseVector[Double]):Double = {
	val newKernelFunc = kernelFunc.changeHyperParams(hyperParams)
	val kernelMatrix = MatrixUtils.buildKernelMatrix(newKernelFunc,trainInput)
	val epParameterEstimator = new EpParameterEstimator(kernelMatrix,targets,stopCriterion)
	val (siteParams,_) = epParameterEstimator.estimateSiteParams
	siteParams.marginalLogLikelihood.get
  }

  def logLikelihood(trainInput:DenseMatrix[Double],targets:DenseVector[Int],
							   hyperParams:DenseVector[Double]):(Double,DenseVector[Double]) = {

	val newKernelFunc = kernelFunc.changeHyperParams(hyperParams)
	val kernelMatrix = MatrixUtils.buildKernelMatrix(newKernelFunc,trainInput)
	val epParameterEstimator = new EpParameterEstimator(kernelMatrix,targets,stopCriterion)
	val (siteParams,lowerTriangular) = epParameterEstimator.estimateSiteParams
	val optimInput = HyperParameterOptimInput(siteParams = siteParams,lowerTriangular = lowerTriangular,
	  kernelMatrix = kernelMatrix,trainInput = trainInput)
	val derivatives = logLikelihoodDerivativesAfterHyperParams(optimInput,newKernelFunc)
	(siteParams.marginalLogLikelihood.get,derivatives)
  }

  def logLikelihoodDerivativesAfterHyperParams(optimInput:HyperParameterOptimInput,
						  kernelFun:KernelFunc):DenseVector[Double] = {
	val (siteParams,lowerTriangular,trainData) =
	  (optimInput.siteParams,optimInput.lowerTriangular,optimInput.trainInput)
	val kernelMatrix:kernelMatrixType = optimInput.kernelMatrix
	val tauDiagVector:DenseVector[Double] = sqrt(siteParams.tauSiteParams)
	val tauDiagMatrix:DenseMatrix[Double] = diag(tauDiagVector)
  	val temp:DenseVector[Double] = backSolve(R = lowerTriangular.t,
	  b = (tauDiagMatrix *  kernelMatrix) * siteParams.niSiteParams)
	val bVector:DenseVector[Double] =
	  siteParams.niSiteParams - forwardSolve(L = tauDiagMatrix * lowerTriangular,b = temp)
	val temp1:DenseMatrix[Double] = forwardSolve(L = lowerTriangular,b = tauDiagMatrix)
	val rMatrix:DenseMatrix[Double] = (bVector * bVector.t)
		- backSolve(R = tauDiagMatrix * lowerTriangular.t,b = temp1)
	(0 until kernelFun.hyperParametersNum).foldLeft(DenseVector.zeros[Double](kernelFun.hyperParametersNum)){
	  case (gradient,index) =>
		val cMatrix:DenseMatrix[Double] = buildKernelDerMatrixAfterHyperParam(trainData,index+1,kernelFun)
		val logLikelihoodDerAfterParam:Double = 0.5*trace(rMatrix*cMatrix)
	    gradient.update(index,logLikelihoodDerAfterParam); gradient
	}
  }

  private def buildKernelDerMatrixAfterHyperParam(trainInput:DenseMatrix[Double],hyperParameterNum:Int
												  ,kernelFunc:KernelFunc
												  ):DenseMatrix[Double] = {

	val func:(DenseVector[Double],DenseVector[Double],Boolean) => Double = {(vec1,vec2,sameIndex) =>
	  kernelFunc.derAfterHyperParam(hyperParameterNum)(vec1,vec2,sameIndex)
	}
	buildMatrixWithFunc(trainInput)(func)
  }

}


object MarginalLikelihoodEvaluator {

  type kernelAfterParamDerivative = (Double,DenseVector[Double],DenseVector[Double]) => Double

  type logLikelihoodAfterParamDerivative = (Double) => Double

  case class HyperParameterOptimInput(siteParams:SiteParams,lowerTriangular:DenseMatrix[Double],
									  kernelMatrix:DenseMatrix[Double],trainInput:DenseMatrix[Double])

}