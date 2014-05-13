package dynamicalsystems.filtering

import org.scalatest.WordSpec
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import utils.StatsUtils.GaussianDistribution
import breeze.linalg.{cholesky, DenseMatrix, DenseVector}
import dynamicalsystems.filtering.UnscentedKalmanFilter.{UnscentedFilteringInput, UnscentedTransformParams}
import dynamicalsystems.filtering.SsmExamples.{KitagawaSsm, SinusoidalSsm}
import dynamicalsystems.filtering.SsmTypeDefinitions.SeriesGenerationData
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner
import org.springframework.test.context.{TestContextManager, ContextConfiguration}
import gp.optimization.GPOptimizer
import org.springframework.beans.factory.annotation.Autowired
import gp.regression.GpPredictor
import dynamicalsystems.tests.SsmTestingUtils

/**
 * Created by mjamroz on 05/04/14.
 */

@RunWith(classOf[SpringJUnit4ClassRunner])
@ContextConfiguration(locations = Array("classpath:config/spring-context.xml"))
class UnscentedKalmanFilterTest extends WordSpec with SsmTestingUtils{

  val sinRNoise:DenseMatrix[Double] = DenseMatrix((0.1*0.1))
  val sinQNoise:DenseMatrix[Double] = sinRNoise
  val kitRNoise:DenseMatrix[Double] = DenseMatrix((0.2*0.2))
  val kitQNoise:DenseMatrix[Double] = DenseMatrix((0.01*0.01))
  val initHiddenStateDistr:GaussianDistribution = GaussianDistribution.standard
  val initHiddenStateDistrKit:GaussianDistribution = GaussianDistribution(mean = DenseVector(0.),
	sigma = DenseMatrix((0.5*0.5)))

  new TestContextManager(this.getClass).prepareTestInstance(this)

  @Autowired
  var gpOptimizer:GPOptimizer = _
  @Autowired
  var gpPredictor:GpPredictor = _

  "unscented transform" should {

	"transform sigma pointSet from 1d gaussian distribution" in {

	  val gaussian1d = GaussianDistribution(mean = DenseVector(2.),sigma = DenseMatrix((2.)))
	  val ukf = new UnscentedKalmanFilter(gpOptimizer)
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
	  val ukf = new UnscentedKalmanFilter(gpOptimizer)
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
	  val (hidden,obs) = generateSinSamples(seqLength)
	  val ukf = new UnscentedKalmanFilter(gpOptimizer)
	  val defaultParams = UnscentedTransformParams().copy(alpha = 1.0)
	  val ukfInput = ukfSinInput(obs,seqLength)

	  val out = ukf.inferHiddenState(ukfInput,Some(defaultParams),true)
	  val out1 = ukf.inferHiddenState(ukfInput,Some(defaultParams.copy(alpha = 2.012,beta = 0.24,kappa = 0.4871)),true)
	  assert(out.hiddenMeans.cols == seqLength)
	  assert(out.hiddenMeans.rows == hidden.rows)
	  assert(out.hiddenCovs.length == seqLength)
	}

	"infer hidden state in kitagawa problem" in {
	  val seqLength = 50
	  val (hidden,obs) = generateKitSamples(seqLength)
	  val ukf = new UnscentedKalmanFilter(gpOptimizer)
	  val defaultParams = UnscentedTransformParams().copy(alpha = 1.0)
	  val ukfInput = ukfKitInput(obs,seqLength)
	  val out = ukf.inferHiddenState(ukfInput,Some(defaultParams),true)
	  val out1 = ukf.inferHiddenState(ukfInput,Some(defaultParams.copy(alpha = 2.012,beta = 0.24,kappa = 0.4871)),true)
	  assert(out.hiddenMeans.cols == seqLength)
	  assert(out.hiddenMeans.rows == hidden.rows)
	  assert(out.hiddenCovs.length == seqLength)
	}
	
	"infer hidden state with unscented transform params optimization in sinusoidal problem" in {
	  val seqLength = 200
	  val (hidden,obs) = generateSinSamples(seqLength)
	  val ukf = new UnscentedKalmanFilter(gpOptimizer)
	  val ukfInput = ukfSinInput(obs,seqLength)
	  val optimizedOut = ukf.inferWithUkfOptimWrtToMarginall(ukfInput,None)
	  val out = ukf.inferHiddenState(ukfInput,None,true)
	  println(s"UKF - Log likelihood = ${out.logLikelihood}")
	  println(s"UKF-L - Log likelihood = ${optimizedOut.logLikelihood}")
	}

	"infer hidden state with unscented transform params optimization in kitagawa problem" in {
	  val seqLength = 200
	  val (hidden,obs) = generateKitSamples(seqLength)
	  val ukf = new UnscentedKalmanFilter(gpOptimizer)
	  val ukfInput = ukfKitInput(obs,seqLength)
	  val bestParamsForKit = UnscentedTransformParams(alpha = 0.38,beta = 1.276,kappa = 2.58)
	  val out = ukf.inferHiddenState(ukfInput,Some(bestParamsForKit),true)
	  val optimizedOut = ukf.inferWithUkfOptimWrtToMarginall(ukfInput,None)
	  println(s"Kit:UKF - Log likelihood = ${out.logLikelihood}")
	  println(s"Kit:UKF-L - Log likelihood = ${optimizedOut.logLikelihood}")
	}

  }

  "gp unscented kalman filter" should {

	"infer hidden state in sinusoidal problem" in {

	  val seqLength = 50
	  val (hidden,obs) = generateSinSamples(seqLength)
	  val gpUkf = new GPUnscentedKalmanFilter(gpOptimizer,gpPredictor)
	  val ukfInput = ukfSinInput(obs,seqLength)
	  val out =  gpUkf.inferHiddenState(ukfInput,None,hidden,true,false)
	  assert(out.hiddenMeans.cols == seqLength)
	  assert(out.hiddenMeans.rows == hidden.rows)
	  assert(out.hiddenCovs.length == seqLength)
	  val optimizedOut = gpUkf.inferHiddenState(ukfInput,None,hidden,true,true)
	  println(s"GP-UKF Log likelihood = ${out.logLikelihood}")
	  println(s"GP-UKF-HPO Log likelihood = ${optimizedOut.logLikelihood}")
	}

  }

  private def generateSinSamples(seqLength:Int) = {
	generateSamples(seqLength,new SinusoidalSsm,(sinRNoise,sinQNoise),initHiddenStateDistr)
  }

  private def generateKitSamples(seqLength:Int) = {
	generateSamples(seqLength,new KitagawaSsm,(kitRNoise,kitQNoise),initHiddenStateDistrKit)
  }

  private def generateSamples(seqLength:Int,ssmSampler:SsmModel,
							  noises:(DenseMatrix[Double],DenseMatrix[Double]),initDistr:GaussianDistribution) = {
	val genData = SeriesGenerationData(qNoise = cloneMatrix(noises._1,seqLength),
	  rNoise = cloneMatrix(noises._2,seqLength),initHiddenState = Right(initDistr))
	ssmSampler.generateSeries(seqLength,genData)
  }
  
  private def ukfSinInput(obs:DenseMatrix[Double],seqLength:Int):UnscentedFilteringInput = {
	ukfInput(obs,new SinusoidalSsm,seqLength,(sinRNoise,sinQNoise),initHiddenStateDistr)
  }

  private def ukfKitInput(obs:DenseMatrix[Double],seqLength:Int):UnscentedFilteringInput = {
	ukfInput(obs,new KitagawaSsm,seqLength,(kitRNoise,kitQNoise),initHiddenStateDistrKit)
  }

  private def ukfInput(obs:DenseMatrix[Double],ssmModel:SsmModel,seqLength:Int,
					   noises:(DenseMatrix[Double],DenseMatrix[Double]),initDistr:GaussianDistribution) = {
	val (qNoiseFunc,rNoiseFunc) = UnscentedFilteringInput.classicUkfNoise(
	  cloneMatrix(noises._2,seqLength),cloneMatrix(noises._1,seqLength))
	UnscentedFilteringInput(ssmModel = ssmModel,observations = obs,
	  u = None,initMean = initDistr.mean,initCov = initDistr.sigma,
	  qNoise = qNoiseFunc,rNoise = rNoiseFunc)
  }

}
