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
	  dv.forall{ case (index,value) => value ~= dv1(index)}
	}
  }

}
