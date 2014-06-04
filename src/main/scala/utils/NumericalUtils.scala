package utils

import breeze.linalg.DenseVector

/**
 * Created by mjamroz on 07/04/14.
 */
object NumericalUtils {

  case class Precision(val p:Double)

  implicit class DoubleWithAlmostEquals(val d:Double) extends AnyVal {
	def ~=(d2:Double)(implicit p:Precision):Boolean = (d - d2).abs < p.p
  }

  implicit class DoubleVectorWithAlmostEquals(val dv:DenseVector[Double]) {
	def ~= (dv1:DenseVector[Double])(implicit p:Precision):Boolean = {
	  dv.forall{ (index:Int,value:Double) => value ~= dv1(index)}
	}
  }

  trait vectorFunc {

	def apply[T <: (Double) => Double](dv:DenseVector[Double])(func:T):DenseVector[Double] = {
	  dv.map(func(_))
	}

  }

  object sin extends vectorFunc {

	def apply(dv:DenseVector[Double]):DenseVector[Double] = {
		super.apply(dv){value:Double => math.sin(value)}
	}

  }

}
