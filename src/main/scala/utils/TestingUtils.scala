package utils

import breeze.linalg.DenseVector

/**
 * Created by mjamroz on 19/04/14.
 */
object TestingUtils {

  class ScalaObjectsCreator{

	def none[T]:Option[T] = None

	def some[T](arg:T):Option[_] = Some(arg)

	def co2HyperParamsVec:DenseVector[Double] = {
	  val hp = DenseVector(60.,70.,8.,50.,2.,0.34,2.4,0.88,0.26,0.2,0.19)
	  hp
	}

	def defaultCo2HyperParamsVec:DenseVector[Double] = {
	  DenseVector(100.,50.,50.,50.,2.,1.,1.,1.,1.,1.,0.5)
	}

  }

}
