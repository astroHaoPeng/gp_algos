package utils

import org.scalatest.WordSpec
import breeze.linalg.{cholesky, DenseMatrix, DenseVector}

/**
 * Created by mjamroz on 30/03/14.
 */
class StatsUtilsTest extends WordSpec {

  import StatsUtils._
  import NumericalUtils._

  implicit val precision = Precision(p = 0.0001)

  "gaussian distribution" should {
	
	"return proper density value" in {
	  
	 	val (mean,covs) = (DenseVector(0.,0.),DenseMatrix((1.,0.),(0.,1.)))
	  	println(gaussianDensity(DenseVector(0.,0.),mean,covs))
	  
	}

  }

  "function computing mean and sigma of normal distribution from data" should {

	"generate proper values" in {

	  val data = DenseMatrix((1.,5.),(0.5,3.),(0.6,4.))
	  val (mean,sigma) = meanAndVarOfData(data)
	  assert(mean ~= DenseVector(0.7,4.))
	  cholesky(sigma)
	}
  }

}
