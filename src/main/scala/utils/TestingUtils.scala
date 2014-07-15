package utils

import breeze.linalg.DenseVector
import utils.KernelRequisites.GaussianRbfParams

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
	  //DenseVector(100.,50.,50.,50.,2.,1.,1.,1.,1.,1.,0.5)
	  lengthScales(11)
	}

	def lengthScales(dim:Int) = DenseVector.ones[Double](dim)

	def optimalBostonHp:GaussianRbfParams = {
	  GaussianRbfParams(-1130.9947925594922,
		DenseVector(566.7442989546967, 735.3624053303566, 536.1791714384265, 610.4651246027757, 626.0353185058663,
		  5.528303239800252, 2853.7974583131668, 1006.5144425910395, 504.78702267976087, 1287.102910849582,
		  387.26286609421436, 2678.0145551405353, 1093.4657445540006),2.1355086421066893)
	}

  }

}
