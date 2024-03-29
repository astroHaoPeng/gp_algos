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

  "mean squared error" should {

	"return proper value" in {

	  val (estimate,trueVals) = (DenseMatrix((1.,2.),(3.,4.),(5.,6.)),DenseMatrix((1.1,2.1),(3.1,4.1),(5.1,6.1)))
	  val mseVal = mse(estimate,trueVals)
	  assert(mseVal ~= 0.02)
	}

  }

}
