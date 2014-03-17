package gp.classification

import utils.KernelRequisites.KernelFuncHyperParams
import breeze.optimize.{LBFGS, DiffFunction}
import breeze.linalg.{DenseMatrix, DenseVector}
import gp.classification.GpClassifier.ClassifierInput

/**
 * Created by mjamroz on 16/03/14.
 */
object HyperParamsOptimization {

  trait HyperParameterOptimizer {

	def optimizeHyperParams(optimizationInput:ClassifierInput):KernelFuncHyperParams

  }

  class BreezeLBFGSOptimizer(marginalLikelihoodEvaluator:MarginalLikelihoodEvaluator) extends HyperParameterOptimizer{

	def optimizeHyperParams(optimizationInput:ClassifierInput): KernelFuncHyperParams = {

	  val (trainData,targets) = (optimizationInput.trainInput,optimizationInput.targets)

	  /*diffFunction will be minimized so it needs to be equal to -logLikelihood*/
	  val diffFunction = new DiffFunction[DenseVector[Double]] {

		def calculate(hyperParams: DenseVector[Double]): (Double, DenseVector[Double]) = {

		  val (logLikelihood,derivatives) = marginalLikelihoodEvaluator.logLikelihood(trainData,targets,hyperParams)
		  assert(hyperParams.length == derivatives.length)
		  val evaluatedDerivatives:DenseVector[Double]  = (0 until derivatives.length).
			foldLeft(DenseVector.zeros[Double](hyperParams.length)){
				case (gradient,index) => gradient.update(index,-derivatives(index)(hyperParams(index))); gradient
		  }
		  (-logLikelihood,evaluatedDerivatives)
		}
	  }

	  val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 10,m = 3)
	  val optimizedParams = lbfgs.minimize(diffFunction,optimizationInput.hyperParams.toDenseVector)
	  optimizationInput.hyperParams.fromDenseVector(optimizedParams)
	}
  }

  case class HyperOptimizationInput(trainInput:DenseMatrix[Double],targets:DenseVector[Int],
									 initParams:KernelFuncHyperParams)
  
}

