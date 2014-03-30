package utils

import org.scalatest.WordSpec
import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Created by mjamroz on 30/03/14.
 */
class StatsUtilsTest extends WordSpec {

  import StatsUtils._

  "gaussian distribution" should {
	
	"return proper density value" in {
	  
	 	val (mean,covs) = (DenseVector(0.,0.),DenseMatrix((1.,0.),(0.,1.)))
	  	println(gaussianDensity(DenseVector(0.,0.),mean,covs))
	  
	}
	
  }

}
