package gp.optimization

import optimization.Optimization
import Optimization._
import breeze.numerics._

/**
 * Created by mjamroz on 19/04/14.
 */
object OptimizationUtils {

  object Functions1D{
	val quadraticFunction: objectiveFunction = {
	  point => -2 * point(0) * point(0) + 6 * point(0) + 5
	}

	val strangeFunc: objectiveFunction = {
	  point =>   cos(log(point(0)*point(0))) * sin(point(0)*point(0))
	}

	val xSinXFunc: objectiveFunction = {
	  point => point(0) + sin(point(0))
	}

	val strangeFunc1: objectiveFunction = {
	  point => val x = point(0); x + sin(0.5*x) - cos(x*x)  /*x + sin(0.5*x) - cos(2.0*x)*/
	}
  }

  object Functions2D{

	val rastriginFunc: objectiveFunction = {
	  point => val (x,y) = (point(0),point(1))
		       20 + x*x + y*y -10*(cos(2*math.Pi*x) + cos(2*math.Pi*y))
	}

  }

}
