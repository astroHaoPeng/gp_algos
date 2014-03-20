package gp.classification

import utils.KernelRequisites.KernelFuncHyperParams
import breeze.optimize.{LBFGS, DiffFunction}
import breeze.linalg.{DenseMatrix, DenseVector}
import gp.classification.GpClassifier.ClassifierInput
import org.apache.commons.math3.analysis.{MultivariateVectorFunction, MultivariateFunction, DifferentiableMultivariateFunction}
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer.Formula
import org.apache.commons.math3.optim._
import org.apache.commons.math3.optim.nonlinear.scalar.{ObjectiveFunctionGradient, ObjectiveFunction, GoalType}
import org.slf4j.LoggerFactory
import scala.Some
import gp.classification.GpClassifier.ClassifierInput

/**
 * Created by mjamroz on 16/03/14.
 */
object HyperParamsOptimization {

  val apacheLogger = LoggerFactory.getLogger(classOf[HyperParamsOptimization.HyperParameterOptimizer])

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
		  apacheLogger.info(s"Current solution is = ${hyperParams}, objective function value = ${-logLikelihood}")
		  (-logLikelihood,evaluatedDerivatives)
		}
	  }

	  val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 10,m = 3)
	  val optimizedParams = lbfgs.minimize(diffFunction,optimizationInput.hyperParams.toDenseVector)
	  optimizationInput.hyperParams.fromDenseVector(optimizedParams)
	}
  }

  class ApacheCommonsOptimizer(marginalLikelihoodEvaluator:MarginalLikelihoodEvaluator) extends HyperParameterOptimizer{

	class IterationLimitConvergenceChecker(val limit:Int) extends ConvergenceChecker[PointValuePair]{

	  def converged(iteration: Int, previous: PointValuePair, current: PointValuePair): Boolean = {
	  	iteration >= limit
	  }
	}

	class LogLikelihoodObjectiveFunction(optimizationInput:ClassifierInput) extends DifferentiableMultivariateFunction{
	  val pointGradientMapping:scala.collection.mutable.Map[Array[Double],(Array[Double],Double)] =
		scala.collection.mutable.Map()

	  def gradient(): MultivariateVectorFunction = {

		new MultivariateVectorFunction {

		  def value(hyperParams: Array[Double]): Array[Double] = {
			pointGradientMapping.get(hyperParams) match {
			  case Some((gradient,value)) => gradient
			  case None =>
				val (logLikelihood,gradient) = computeValueAndGradient(optimizationInput,hyperParams)
				apacheLogger.info(s"Current solution is = ${DenseVector(hyperParams)}, objective function value = ${logLikelihood}")
				pointGradientMapping.put(hyperParams,(gradient,logLikelihood)); gradient
			}
		  }
		}
	  }

	  def value(hyperParams: Array[Double]): Double = {

		pointGradientMapping.get(hyperParams) match {
		  case Some((_,logLikelihood)) => logLikelihood
		  case None =>
			val (logLikelihood,derivatives) = computeValueAndGradient(optimizationInput,hyperParams)
			pointGradientMapping.put(hyperParams,(derivatives,logLikelihood))
			logLikelihood
		}
	  }

	  def partialDerivative(k: Int): MultivariateFunction = {

		new MultivariateFunction {

		  def value(hyperParams: Array[Double]): Double = {
			pointGradientMapping.get(hyperParams) match {
			  case Some((gradient,value)) => gradient(k)
			  case None =>
				val (logLikelihood,gradient) = computeValueAndGradient(optimizationInput,hyperParams)
				pointGradientMapping.put(hyperParams,(gradient,logLikelihood)); gradient(k)
			}
		  }
		}
	  }
	}

	def optimizeHyperParams(optimizationInput: ClassifierInput): KernelFuncHyperParams = {

	  val objectiveFunction = new LogLikelihoodObjectiveFunction(optimizationInput)
	  val conjugateGradientOptimizer = new NonLinearConjugateGradientOptimizer(
		Formula.POLAK_RIBIERE,new IterationLimitConvergenceChecker(5))
	  val pointValueOptimResult =
		conjugateGradientOptimizer.optimize(new ObjectiveFunction(objectiveFunction),GoalType.MAXIMIZE
		  ,new MaxIter(10),new MaxEval(20),new InitialGuess(optimizationInput.hyperParams.toDenseVector.data),
		new ObjectiveFunctionGradient(objectiveFunction.gradient()))   //optimizationInput.hyperParams.toDenseVector.data)

	  apacheLogger.info(s"Optimal solution is = ${pointValueOptimResult.getPoint}, objective function value = ${pointValueOptimResult.getValue}")
	  optimizationInput.hyperParams.fromDenseVector(DenseVector(pointValueOptimResult.getPoint))
	}

	private def computeValueAndGradient(optimizationInput: ClassifierInput,hyperParams:Array[Double]):(Double,Array[Double])
		 = {

	  val (trainData,targets) = (optimizationInput.trainInput,optimizationInput.targets)
	  val (logLikelihood,derivatives) = marginalLikelihoodEvaluator.logLikelihood(trainData,targets,DenseVector(hyperParams))
	  assert(hyperParams.length == derivatives.length)
	  val evaluatedDerivatives:DenseVector[Double]  = (0 until derivatives.length).
		foldLeft(DenseVector.zeros[Double](hyperParams.length)){
		case (gradient,index) => gradient.update(index,derivatives(index)(hyperParams(index))); gradient
	  }
	  (logLikelihood,evaluatedDerivatives.data)
	}

  }

  case class HyperOptimizationInput(trainInput:DenseMatrix[Double],targets:DenseVector[Int],
									 initParams:KernelFuncHyperParams)
  
}

