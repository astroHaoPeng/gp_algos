package gp.regression

import breeze.linalg.{DenseVector, DenseMatrix}
import gp.regression.GpPredictor.PredictionInput
import utils.KernelRequisites.{KernelFunc, KernelFuncHyperParams}
import utils.StatsUtils
import org.springframework.context.support.GenericXmlApplicationContext
import gp.regression.Co2Prediction.Co2Kernel
import svm.SvmRegressionImpl

/**
 * Created by mjamroz on 17/08/14.
 */
class Co2PredictionExecutor {

  case class PredictionTestInstance(predictionInput:PredictionInput,testTargets:DenseVector[Double])
  case class PredictionTestSuiteResult(testTargets:DenseVector[Double],predictedValues:DenseVector[Double],
										mse:Double,ll:Option[Double]){

	override def toString = {
	  val tempBuffer = (0 until testTargets.length).foldLeft(new StringBuffer("")){
		case (buff,index) =>
		  val line = s"$index: predicted value: ${predictedValues(index)}, true value: ${testTargets(index)}"
		  buff.append(line)
		  buff.append('\n')
	  }
	  val summarization = s"[mean squared error = $mse}, rmse = ${math.sqrt(mse)}, log likelihood = ${ll}]"
	  tempBuffer.append(summarization).toString
	}

  }

  type testExecutingFunc = (PredictionTestInstance,KernelFunc) => (DenseVector[Double],Option[Double])

  val genericAppContext = new GenericXmlApplicationContext()
  genericAppContext.load("classpath:config/spring-context.xml")
  genericAppContext.refresh()

  val gpMeanPredictor:testExecutingFunc = {
	(testInstance,kernel) =>
	  val gpPredictor = new GpPredictor(kernel)
	  val (posterior,ll) = gpPredictor.predict(testInstance.predictionInput)
	  (posterior.mean,Some(ll))
  }

  val gpSamplePredictor:testExecutingFunc = {
	(testInstance,kernel) =>
	  val gpPredictor = new GpPredictor(kernel)
	  val (posterior,ll) = gpPredictor.predict(testInstance.predictionInput)
	  (StatsUtils.NormalDistributionSampler.sample(posterior),Some(ll))
  }

  val svmPredictor:testExecutingFunc = {
	(testInstance,kernel) =>
	  val svmPredictor = new SvmRegressionImpl(kernel)
	  (svmPredictor.predict(testInstance.predictionInput),None)
  }

  def executeTestForPredictors(trainTestRatio:Double):
  	(PredictionTestSuiteResult,PredictionTestSuiteResult,PredictionTestSuiteResult) = {
	val testInstance = prepareTestInstance(trainTestRatio)
	val kernel = genericAppContext.getBean("co2Kernel",classOf[Co2Kernel])
	val optimalHps = obtainOptimalHyperParamsForKernel(testInstance.predictionInput,kernel)
	val optimalKernel = kernel.changeHyperParams(optimalHps.toDenseVector)
	val gpMeanResult = executeTest(testInstance,optimalKernel)(gpMeanPredictor)
	val gpSampleResult = executeTest(testInstance,optimalKernel)(gpSamplePredictor)
	val svmResult = executeTest(testInstance,kernel)(svmPredictor)
	(gpMeanResult,gpSampleResult,svmResult)
  }

  def executeTest(testInstance:PredictionTestInstance,kernel:KernelFunc)(executingFunc:testExecutingFunc):PredictionTestSuiteResult = {
	val (predictedValues,ll) = executingFunc(testInstance,kernel)
	val trueValues = testInstance.testTargets
	val mseVal = StatsUtils.mse(predictedValues.toDenseMatrix,trueValues.toDenseMatrix,horSample = true)
	PredictionTestSuiteResult(testTargets = testInstance.testTargets,
	  predictedValues = predictedValues,mse = mseVal,ll = ll)
  }

  def prepareTestInstance(trainTestRatio:Double):PredictionTestInstance = {

	val co2Data:DenseMatrix[Double] = Co2Prediction.loadInput(fileName = "co2/maunaLoa2D.txt",colNum = 2)

	val wholeDsTargets:DenseVector[Double] = co2Data(::,1)
	val wholeDs:DenseMatrix[Double] = co2Data(::,0).toDenseMatrix.t
	val sampleNum = co2Data.rows
	val trainNum = (trainTestRatio*sampleNum).toInt
	assert(wholeDs.rows == sampleNum)
	assert(wholeDs.cols == 1)
	val orderedIds:IndexedSeq[Int] = 0 until co2Data.rows
	val shuffledIds:IndexedSeq[Int] = util.Random.shuffle(orderedIds)
	val shuffledTargets:DenseVector[Double] = DenseVector.tabulate[Double](sampleNum){
	  case id => wholeDsTargets(shuffledIds(id))}
	val shuffledDs:DenseMatrix[Double] = DenseMatrix.tabulate[Double](sampleNum,1){
	  case (row,_) => wholeDs(shuffledIds(row),0)
	}
	val predInput = PredictionInput(trainingData = shuffledDs(0 until trainNum,::),
	  testData = shuffledDs(trainNum until sampleNum,::),sigmaNoise = None,targets = shuffledTargets(0 until trainNum))
	PredictionTestInstance(predictionInput = predInput,testTargets = shuffledTargets(trainNum until sampleNum))
  }

  def obtainOptimalHyperParamsForKernel(predictionInput:PredictionInput,kernel:KernelFunc):KernelFuncHyperParams = {
	val samplePredictor = new GpPredictor(kernel)
	val (trainingData,targets) = (predictionInput.trainingData,predictionInput.targets)
	samplePredictor.obtainOptimalHyperParams(trainingData,None,targets,optimizeNoise = true)
  }

}
