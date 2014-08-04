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

  import optimization.Optimization._

  val apacheLogger = LoggerFactory.getLogger(classOf[HyperParamsOptimization.HyperParameterOptimizer])

  trait HyperParameterOptimizer {

	def optimizeHyperParams(optimizationInput:ClassifierInput): KernelFuncHyperParams
  }


  class GradientHyperParamsOptimizer(marginalLikelihoodEvaluator:MarginalLikelihoodEvaluator,
									 gradOptimizer:GradientBasedOptimizer)
		extends HyperParameterOptimizer {

	def optimizeHyperParams(optimizationInput:ClassifierInput): KernelFuncHyperParams = {

	  val targets = optimizationInput.targets
	  val funcWithGradient:objectiveFunctionWithGradient = {hyperParams:Array[Double] =>
		val (logLikelihood,derivatives) =
		  marginalLikelihoodEvaluator.logLikelihood(optimizationInput.trainData.get,
			targets,DenseVector(hyperParams))
		assert(hyperParams.length == derivatives.length)
		apacheLogger.info(s"Current solution is = ${hyperParams}, objective function value = ${-logLikelihood}")
		(logLikelihood,derivatives.toArray)
	  }
	  /*
	  val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 10,m = 3)
	  val optimizedParams = lbfgs.minimize(diffFunction,optimizationInput.initHyperParams.toDenseVector) */
	  val optimizedParams = gradOptimizer.maximize(funcWithGradient,
		optimizationInput.initHyperParams.toDenseVector.toArray)

	  optimizationInput.initHyperParams.fromDenseVector(DenseVector(optimizedParams))
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
		  ,new MaxIter(10),new MaxEval(20),new InitialGuess(optimizationInput.initHyperParams.toDenseVector.data),
		new ObjectiveFunctionGradient(objectiveFunction.gradient()))   //optimizationInput.hyperParams.toDenseVector.data)

	  apacheLogger.info(s"Optimal solution is = ${pointValueOptimResult.getPoint}, objective function value = ${pointValueOptimResult.getValue}")
	  optimizationInput.initHyperParams.fromDenseVector(DenseVector(pointValueOptimResult.getPoint))
	}

	private def computeValueAndGradient(optimizationInput: ClassifierInput,hyperParams:Array[Double]):(Double,Array[Double])
		 = {

	  val targets = optimizationInput.targets
	  val (logLikelihood,derivatives) = marginalLikelihoodEvaluator.logLikelihood(
		optimizationInput.trainData.get,targets,DenseVector(hyperParams))
	  (logLikelihood,derivatives.data)
	}

  }

  case class HyperOptimizationInput(trainInput:DenseMatrix[Double],targets:DenseVector[Int],
									 initParams:KernelFuncHyperParams)
  
}

