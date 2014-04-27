package dynamicalsystems.filtering

import org.scalatest.WordSpec
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import utils.StatsUtils.GaussianDistribution
import breeze.linalg.{cholesky, DenseMatrix, DenseVector}
import dynamicalsystems.filtering.UnscentedKalmanFilter.{UnscentedFilteringInput, UnscentedTransformParams}
import dynamicalsystems.filtering.SsmExamples.SinusoidalSsm
import dynamicalsystems.filtering.SsmTypeDefinitions.SeriesGenerationData

/**
 * Created by mjamroz on 05/04/14.
 */

@RunWith(classOf[JUnitRunner])
class UnscentedKalmanFilterTest extends WordSpec with SsmTestingUtils{

  val rNoise:DenseMatrix[Double] = DenseMatrix((0.1*0.1))
  val qNoise:DenseMatrix[Double] = rNoise
  val initHiddenStateDistr:GaussianDistribution = GaussianDistribution.standard

  "unscented transform" should {

	"transform sigma pointSet from 1d gaussian distribution" in {

	  val gaussian1d = GaussianDistribution(mean = DenseVector(2.),sigma = DenseMatrix((2.)))
	  val ukf = new UnscentedKalmanFilter
	  val defaultParams = UnscentedTransformParams()
	  val transform = ukf.unscentedTransform(gaussian1d,defaultParams){vec => vec}
	  val transformedDistr:GaussianDistribution = transform.distribution
	  assert(transformedDistr.dim == 1)
	  assert(transformedDistr.sigma(0,0) != 0.0)
	  assert(transform.sigmaPoints.rows == 3)
	  assert(transform.transformedSigmaPoints.rows == 3)
	  assert(transform.transformedSigmaPoints == transform.sigmaPoints)
	}

	"transform sigma pointSet from 3d gaussian distribution" in {

	  val gaussian3d = GaussianDistribution(mean = DenseVector(1.,2.,3.),sigma = DenseMatrix((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
	  cholesky(gaussian3d.sigma)
	  val ukf = new UnscentedKalmanFilter
	  val defaultParams = UnscentedTransformParams()
	  val transform = ukf.unscentedTransform(gaussian3d,defaultParams){vec => vec :* vec}
	  val transformedDistr:GaussianDistribution = transform.distribution
	  assert(transformedDistr.dim == 3)
	  assert(transformedDistr.sigma.rows == 3 && transformedDistr.sigma.cols == 3)
	  assert(transform.sigmaPoints.rows == 7)
	  assert(transform.transformedSigmaPoints.rows == 7)
	  cholesky(transformedDistr.sigma)
	}

  }

  "unscented kalman filter" should {

	"infer hidden state in sinusoidal problem" in {

	  val seqLength = 50
	  val (hidden,obs) = generateSamples(seqLength)
	  val ukf = new UnscentedKalmanFilter
	  val defaultParams = UnscentedTransformParams().copy(alpha = 1.0)
	  val ukfInput = UnscentedFilteringInput(ssmModel = new SinusoidalSsm,observations = obs,
		u = None,initMean = initHiddenStateDistr.mean,initCov = initHiddenStateDistr.sigma,
		qNoise = cloneMatrix(rNoise,seqLength),rNoise = cloneMatrix(rNoise,seqLength))
	  val out = ukf.inferHiddenState(ukfInput,Some(defaultParams),true)
	  val out1 = ukf.inferHiddenState(ukfInput,Some(defaultParams.copy(alpha = 0.68)),true)
	  assert(out.hiddenMeans.cols == seqLength)
	  assert(out.hiddenMeans.rows == hidden.rows)
	  assert(out.hiddenCovs.length == seqLength)
	}

  }

  private def generateSamples(seqLength:Int) = {

	val genData = SeriesGenerationData(qNoise = cloneMatrix(rNoise,seqLength),
	  rNoise = cloneMatrix(qNoise,seqLength),initHiddenState = Right(initHiddenStateDistr))
	val ssmSampler = new SinusoidalSsm
	ssmSampler.generateSeries(seqLength,genData)

  }

}
