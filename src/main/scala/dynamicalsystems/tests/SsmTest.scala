package dynamicalsystems.tests

import breeze.linalg.{DenseVector, DenseMatrix}
import org.springframework.context.support.GenericXmlApplicationContext
import gp.regression.GpPredictor
import gp.optimization.GPOptimizer
import dynamicalsystems.filtering.SsmTypeDefinitions.SeriesGenerationData
import dynamicalsystems.filtering.UnscentedKalmanFilter.{UnscentedTransformParams, UnscentedFilteringInput}
import dynamicalsystems.filtering.{SsmModel, UnscentedKalmanFilter}
import dynamicalsystems.filtering.KalmanFilter.FilteringOutput
import java.io.File
import utils.IOUtilities

/**
 * Created by mjamroz on 07/05/14.
 */
trait SsmTest extends SsmTestingUtils{

  import utils.StatsUtils._

  val ssmModel:SsmModel
  val rNoise:DenseMatrix[Double]
  val qNoise:DenseMatrix[Double]
  val resultFilePath:String

  val initHiddenStateDistr:GaussianDistribution = GaussianDistribution.standard

  val genericContext = new GenericXmlApplicationContext
  genericContext.load("classpath:config/spring-context.xml")
  genericContext.refresh

  val gpPredictor = genericContext.getBean(classOf[GpPredictor])
  val gpOptimizer = genericContext.getBean(classOf[GPOptimizer])


  def run(seqLength:Int) = {
	val ssmResult = doTheTestWithUkf(seqLength)
	val optimSsmResult = doTheTestWithUkfParamsOptim(seqLength)
	println(s"${this}-UKF-Test: Log likelihood = ${ssmResult.ll}, MSE = ${ssmResult.mse}")
	println(s"${this}-UKF-L-Test: Log likelihood = ${optimSsmResult.ll}, MSE = ${optimSsmResult.mse}")
  }

  def doTheTestWithUkfParamsOptim(seqLength:Int):SsmTestingResult = {
	val func:(UnscentedKalmanFilter,UnscentedFilteringInput) => FilteringOutput = {
	  (ukf,ukfInput) => ukf.inferWithParamOptimization(ukfInput,None)
	}
	testWithUkf(seqLength,("true_optim.dat","predicted_optim.dat"))(func)
  }

  def doTheTestWithUkf(seqLength:Int,
					   params:UnscentedTransformParams = UnscentedTransformParams.defaultParams):SsmTestingResult = {

	val func:(UnscentedKalmanFilter,UnscentedFilteringInput) => FilteringOutput = {
	  (ukf,ukfInput) => ukf.inferHiddenState(ukfInput,Some(params),true)
	}
	testWithUkf(seqLength,("true.dat","predicted.dat"))(func)
  }

  def testWithUkf(seqLength:Int,paths:(String,String))(func:(UnscentedKalmanFilter,UnscentedFilteringInput) => FilteringOutput) = {
	val (hidden,obs) = generateSamples(seqLength)
	val ukf = new UnscentedKalmanFilter(gpOptimizer)
	val ukfInput = ukfInputGen(obs,seqLength)

	val out = func(ukf,ukfInput)
	require(out.logLikelihood.isDefined,"Log likelihood must be computed")
	val gaussianSeq = filteringOutputToNormDistr(out)
	val mseVal:Double = mse(out.hiddenMeans,hidden,horSample = false)
	writeToFile(out,hidden,("true.dat","predicted.dat"))
	SsmTestingResult(hiddenStates = gaussianSeq,ll = out.logLikelihood.get,mse = mseVal)
  }

  def writeToFile(out:FilteringOutput,trueHiddenStates:DenseMatrix[Double],paths:(String,String)) = {
	require(out.hiddenMeans.rows == 1,
	  "Hidden state space dimensionality should be equal to 1 in order to write it to a file")
  	val (trueHiddenFile:File,predictedFile:File) = (new File(s"$resourcePathPrefix/$resultFilePath/${paths._1}"),
	  new File(s"$resourcePathPrefix/$resultFilePath/${paths._2}"))
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
	val genData = SeriesGenerationData(qNoise = cloneMatrix(qNoise,seqLength),
	  rNoise = cloneMatrix(rNoise,seqLength),initHiddenState = Right(initHiddenStateDistr))
	ssmModel.generateSeries(seqLength,genData)
  }

  private def ukfInputGen(obs:DenseMatrix[Double],seqLength:Int):UnscentedFilteringInput = {
	val (qNoiseFunc,rNoiseFunc) = UnscentedFilteringInput.classicUkfNoise(
	  cloneMatrix(qNoise,seqLength),cloneMatrix(rNoise,seqLength))
	UnscentedFilteringInput(ssmModel = ssmModel,observations = obs,
	  u = None,initMean = initHiddenStateDistr.mean,initCov = initHiddenStateDistr.sigma,
	  qNoise = qNoiseFunc,rNoise = rNoiseFunc)
  }

}

