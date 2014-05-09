package dynamicalsystems.tests

import breeze.linalg.{DenseVector, DenseMatrix}
import utils.StatsUtils.GaussianDistribution
import org.springframework.context.support.GenericXmlApplicationContext
import gp.regression.GpPredictor
import gp.optimization.GPOptimizer
import dynamicalsystems.filtering.SsmTypeDefinitions.SeriesGenerationData
import dynamicalsystems.filtering.SsmExamples.SinusoidalSsm
import dynamicalsystems.filtering.UnscentedKalmanFilter.{UnscentedTransformParams, UnscentedFilteringInput}
import dynamicalsystems.filtering.UnscentedKalmanFilter
import dynamicalsystems.tests.SinusoidalSsmTest.SsmTestingResult
import dynamicalsystems.filtering.KalmanFilter.FilteringOutput
import java.io.File
import org.springframework.core.io.ClassPathResource
import utils.IOUtilities

/**
 * Created by mjamroz on 07/05/14.
 */
class SinusoidalSsmTest extends SsmTestingUtils{

  import utils.StatsUtils._

  val rNoise:DenseMatrix[Double] = DenseMatrix((0.1*0.1))
  val qNoise:DenseMatrix[Double] = rNoise
  val initHiddenStateDistr:GaussianDistribution = GaussianDistribution.standard

  val genericContext = new GenericXmlApplicationContext
  genericContext.load("classpath:config/spring-context.xml")
  genericContext.refresh

  val gpPredictor = genericContext.getBean(classOf[GpPredictor])
  val gpOptimizer = genericContext.getBean(classOf[GPOptimizer])


  def doTheTestWithUkf(seqLength:Int,
					   params:UnscentedTransformParams = UnscentedTransformParams.defaultParams):SsmTestingResult = {

	val (hidden,obs) = generateSamples(seqLength)
	val ukf = new UnscentedKalmanFilter(gpOptimizer)
	val ukfInput = ukfSinInput(obs,seqLength)

	val out = ukf.inferHiddenState(ukfInput,Some(params),true)
	require(out.logLikelihood.isDefined,"Log likelihood must be computed")
	val gaussianSeq = filteringOutputToNormDistr(out)
	val mseVal:Double = mse(out.hiddenMeans,hidden,horSample = false)
	writeToFile(out,hidden)
	SsmTestingResult(hiddenStates = gaussianSeq,ll = out.logLikelihood.get,mse = mseVal)
  }

  def writeToFile(out:FilteringOutput,trueHiddenStates:DenseMatrix[Double]) = {
	require(out.hiddenMeans.rows == 1,
	  "Hidden state space dimensionality should be equal to 1 in order to write it to a file")
  	val (trueHiddenFile:File,predictedFile:File) = (new File(s"$resourcePathPrefix/ssm/ukf/true.dat"),
	  new File(s"$resourcePathPrefix/ssm/ukf/predicted.dat"))
	val timeSeries = DenseVector((1 to out.hiddenMeans.cols).toArray)
	val (predicted,trueHidden) = (out.hiddenMeans(0,::).toDenseVector,trueHiddenStates(0,::).toDenseVector)
	val covs = DenseVector(out.hiddenCovs.map(_(0,0)))
	IOUtilities.writeVectorsToFile(trueHiddenFile,timeSeries,trueHidden)
	IOUtilities.writeVectorsToFile(predictedFile,timeSeries,predicted,covs)
  }

  private def filteringOutputToNormDistr(out:FilteringOutput):Seq[GaussianDistribution] = {
	val (means,covs) = (out.hiddenMeans,out.hiddenCovs)
	(0 until covs.length).foldLeft(Seq.empty[GaussianDistribution]){
		case (currentSeq,index) =>
	  		currentSeq :+ GaussianDistribution(mean = means(::,index),sigma = covs(index))
	}
  }

  private def generateSamples(seqLength:Int) = {

	val genData = SeriesGenerationData(qNoise = cloneMatrix(rNoise,seqLength),
	  rNoise = cloneMatrix(qNoise,seqLength),initHiddenState = Right(initHiddenStateDistr))
	val ssmSampler = new SinusoidalSsm
	ssmSampler.generateSeries(seqLength,genData)

  }

  private def ukfSinInput(obs:DenseMatrix[Double],seqLength:Int):UnscentedFilteringInput = {
	val (qNoiseFunc,rNoiseFunc) = UnscentedFilteringInput.classicUkfNoise(
	  cloneMatrix(qNoise,seqLength),cloneMatrix(rNoise,seqLength))
	UnscentedFilteringInput(ssmModel = new SinusoidalSsm,observations = obs,
	  u = None,initMean = initHiddenStateDistr.mean,initCov = initHiddenStateDistr.sigma,
	  qNoise = qNoiseFunc,rNoise = rNoiseFunc)
  }

}

object SinusoidalSsmTest {

  case class SsmTestingResult(hiddenStates:Seq[GaussianDistribution],ll:Double,mse:Double)

  def main(args:Array[String]) = {

	val ssmTest = new SinusoidalSsmTest
	val ssmResult = ssmTest.doTheTestWithUkf(1000)
	println(s"Sinusoidal-UKF-Test: Log likelihood = ${ssmResult.ll}, MSE = ${ssmResult.mse}")

  }

}
