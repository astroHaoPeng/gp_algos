package dynamicalsystems.filtering

import org.scalatest.WordSpec
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import utils.StatsUtils.GaussianDistribution
import breeze.linalg.{cholesky, DenseMatrix, DenseVector}
import dynamicalsystems.filtering.UnscentedKalmanFilter.UnscentedTransformParams

/**
 * Created by mjamroz on 05/04/14.
 */

@RunWith(classOf[JUnitRunner])
class UnscentedKalmanFilterTest extends WordSpec {

  "unscented transform" should {

	"transform sigma pointSet from 1d gaussian distribution" in {

	  val gaussian1d = GaussianDistribution(mean = DenseVector(2.),sigma = DenseMatrix((2.)))
	  val ukf = new UnscentedKalmanFilter
	  val defaultParams = UnscentedTransformParams()
	  val transformedDistr:GaussianDistribution = ukf.unscentedTransform(gaussian1d,defaultParams){vec => vec}
	  assert(transformedDistr.dim == 1)
	  assert(transformedDistr.sigma(0,0) != 0.0)
	}

	"transform sigma pointSet from 3d gaussian distribution" in {

	  val gaussian3d = GaussianDistribution(mean = DenseVector(1.,2.,3.),sigma = DenseMatrix((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
	  cholesky(gaussian3d.sigma)
	  val ukf = new UnscentedKalmanFilter
	  val defaultParams = UnscentedTransformParams()
	  val transformedDistr:GaussianDistribution = ukf.unscentedTransform(gaussian3d,defaultParams){vec => vec :* vec}
	  assert(transformedDistr.dim == 3)
	  assert(transformedDistr.sigma.rows == 3 && transformedDistr.sigma.cols == 3)
	  cholesky(transformedDistr.sigma)
	}

  }

}
