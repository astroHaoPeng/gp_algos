package gp.optimization

/**
 * Created by mjamroz on 14/04/14.
 */
object Optimization {

  type objectiveFunction = (Array[Double] => Double)

  trait Optimizer[Out,Params] {

	def minimize(func:objectiveFunction,params:Params):Out

	def maximize(func:objectiveFunction,params:Params):Out

  }

}
