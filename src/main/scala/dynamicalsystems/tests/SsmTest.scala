package dynamicalsystems.tests

import breeze.linalg.{DenseVector, DenseMatrix}
import org.springframework.context.support.GenericXmlApplicationContext
import gp.regression.GpPredictor
import gp.optimization.GPOptimizer
import dynamicalsystems.filtering.SsmTypeDefinitions.SeriesGenerationData
import dynamicalsystems.filtering.UnscentedKalmanFilter.{UnscentedTransformParams, UnscentedFilteringInput}
import dynamicalsystems.filtering.{GPUnscentedKalmanFilter, SsmModel, UnscentedKalmanFilter}
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
	val samples = generateSamples(seqLength)
	val ssmResult = doTheTestWithUkf(samples)
	val optimSsmResult = doTheTestWithUkfParamsOptim(samples)
	val gpSsmResult = doTheTestWithGpUkf(samples)
	println(s"${this}-UKF-Test: Log likelihood = ${ssmResult.ll}, MSE = ${ssmResult.mse}, NLL = ${ssmResult.nll}")
	println(s"${this}-UKF-L-Test: Log likelihood = ${optimSsmResult.ll}, MSE = ${optimSsmResult.mse}, NLL = ${optimSsmResult.nll}")
	println(s"${this}-GP-UKF: Log likelihood = ${gpSsmResult.ll}, MSE = ${gpSsmResult.mse}, NLL = ${gpSsmResult.nll}")
  }

  def doTheTestWithUkfParamsOptim(samples:(DenseMatrix[Double],DenseMatrix[Double])):SsmTestingResult = {
	val func:(UnscentedKalmanFilter,UnscentedFilteringInput,DenseMatrix[Double]) => FilteringOutput = {
	  (ukf,ukfInput,hidden) => ukf.inferWithUkfOptimWrtToNll(ukfInput,None,hidden)
	}
	testWithUkf(samples,("ukf/true_optim.dat","ukf/predicted_optim.dat"))(func)
  }

  def doTheTestWithUkf(samples:(DenseMatrix[Double],DenseMatrix[Double]),
					   params:UnscentedTransformParams = UnscentedTransformParams.defaultParams):SsmTestingResult = {

	val func:(UnscentedKalmanFilter,UnscentedFilteringInput,DenseMatrix[Double]) => FilteringOutput = {
	  (ukf,ukfInput,_) => ukf.inferHiddenState(ukfInput,Some(params),true)
	}
	testWithUkf(samples,("ukf/true.dat","ukf/predicted.dat"))(func)
  }

  def doTheTestWithGpUkf(samples:(DenseMatrix[Double],DenseMatrix[Double]),
						 params:UnscentedTransformParams = UnscentedTransformParams.defaultParams):SsmTestingResult = {
	testWithGpUkf(samples,("gpukf/true.dat","gpukf/predicted.dat"))
  }

  def evaluateUkfParamsMesh(seqLength:Int):Unit = {
	val (hidden,obs) = generateSamples(seqLength)
	val evaluator = new MeshUkfParamsEvaluator
	val range = Range.Double(0.1,10.,0.5)
	val ranges = IndexedSeq(range,range,range)
	val (ukf,ukfInput) = (new UnscentedKalmanFilter(gpOptimizer),ukfInputGen(obs,seqLength))
	val meshEvalFunction = { ukfParams:UnscentedTransformParams =>
		ukf.inferHiddenState(ukfInput,Some(ukfParams),true).logLikelihood.get
	}
	val meshValues = evaluator.evaluate(ranges)(meshEvalFunction)
	val file = new File(s"$resourcePathPrefix/$resultFilePath/mesh.dat")
	meshValues.writeToFile(file)
  }

  def testWithUkf(samples:(DenseMatrix[Double],DenseMatrix[Double]),paths:(String,String))(
	func:(UnscentedKalmanFilter,UnscentedFilteringInput,DenseMatrix[Double]) => FilteringOutput) = {
	val (hidden,obs) = samples
	val ukf = new UnscentedKalmanFilter(gpOptimizer)
	val ukfInput = ukfInputGen(obs,hidden.cols)

	val out = func(ukf,ukfInput,hidden)
	require(out.logLikelihood.isDefined,"Log likelihood must be computed")
	val gaussianSeq = filteringOutputToNormDistr(out)
	val mseVal:Double = mse(out.hiddenMeans,hidden,horSample = false)
	writeToFile(out,hidden,(paths._1,paths._2))
	val nll = nllOfHiddenData(hidden,gaussianSeq.toArray)
	SsmTestingResult(hiddenStates = gaussianSeq,ll = out.logLikelihood.get,mse = mseVal,nll = nll)
  }

  def testWithGpUkf(samples:(DenseMatrix[Double],DenseMatrix[Double]),paths:(String,String)) = {
	val (hidden,obs) = samples
	val gpUkf = new GPUnscentedKalmanFilter(gpOptimizer,gpPredictor)
	val ukfInput = ukfInputGen(obs,hidden.cols)
	val out = gpUkf.inferHiddenState(ukfInput,None,hidden,true,true)
	require(out.logLikelihood.isDefined,"Log likelihood must be defined")
	val gaussianSeq = filteringOutputToNormDistr(out)
	val mseVal:Double = mse(out.hiddenMeans,hidden,horSample = false)
	writeToFile(out,hidden,paths)
	val nll = nllOfHiddenData(hidden,gaussianSeq.toArray)
	SsmTestingResult(hiddenStates = gaussianSeq,ll = out.logLikelihood.get,mse = mseVal,nll = nll)
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

