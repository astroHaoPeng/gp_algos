package optimization

import breeze.optimize.{LBFGS, DiffFunction}
import breeze.linalg.DenseVector

/**
 * Created by mjamroz on 14/04/14.
 */
object Optimization {

  type objectiveFunction = (Array[Double] => Double)
  type objectiveFunctionWithGradient = (Array[Double] => (Double,Array[Double]))

  trait Optimizer {

	def minimize(func:objectiveFunction,initPoint:Array[Double]):Array[Double]

	def maximize(func:objectiveFunction,initPoint:Array[Double]):Array[Double]

  }

  trait GradientBasedOptimizer{

   	def minimize(func:objectiveFunctionWithGradient,initPoint:Array[Double]):Array[Double]

	def maximize(func:objectiveFunctionWithGradient,initPoint:Array[Double]):Array[Double]

  }

  class BreezeLbfgsOptimizer(maxIter:Int) extends GradientBasedOptimizer {

	def this() = this(10)

	val lbfgsFactory:() => LBFGS[DenseVector[Double]] =
	  {() => new LBFGS[DenseVector[Double]](maxIter = maxIter,m = 4)}

	override def minimize(func:objectiveFunctionWithGradient,initPoint:Array[Double]) = {

	  var (minimumPoint:DenseVector[Double],minimumVal) = (DenseVector(initPoint),Double.MaxValue)
	  val diffFunction = new DiffFunction[DenseVector[Double]] {
		override def calculate(point: DenseVector[Double]): (Double, DenseVector[Double]) = {
		  val (value,gradient) = func(point.toArray)
		  println(s"Func value = $value")
		  if (value < minimumVal){
			minimumPoint = point; minimumVal = value
		  }
		  (value,DenseVector(gradient))
		}
	  }
	  val optimizer = lbfgsFactory()
	  val optimalPoint = optimizer.minimize(diffFunction,DenseVector(initPoint)).toArray
	  val optimalVal = func(optimalPoint)._1
	  if (optimalVal < minimumVal){optimalPoint} else {
		minimumPoint.toArray
	  }
	}

	override def maximize(func:objectiveFunctionWithGradient,initPoint:Array[Double]) = {
	  val minusFunc:objectiveFunctionWithGradient = {point => val (value,grad) = func(point); (-value,grad.map((-1.)*_))}
	  minimize(minusFunc,initPoint)
	}

  }
}
