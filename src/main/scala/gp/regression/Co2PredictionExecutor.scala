package gp.regression

import breeze.linalg.{DenseVector, DenseMatrix}
import gp.regression.GpPredictor.PredictionInput
import utils.KernelRequisites.{GaussianRbfKernel, KernelFunc, KernelFuncHyperParams}
import utils.{DataSetsTestingUtils, TestingUtils, StatsUtils}
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
	  val meanFrom10Samples:DenseVector[Double] = ((1 to 9).foldLeft(StatsUtils.NormalDistributionSampler.sample(posterior)){
		case (acc,_) => acc :+ StatsUtils.NormalDistributionSampler.sample(posterior)}) :/ 10.
	  (meanFrom10Samples,Some(ll))
  }

  val svmPredictor:testExecutingFunc = {
	(testInstance,kernel) =>
	  val svmPredictor = new SvmRegressionImpl(kernel)
	  (svmPredictor.predict(testInstance.predictionInput),None)
  }

  def executeTestForSeKernel(trainTestRatio: Double, fileName: String = "co2/maunaLoa2D.txt"): PredictionTestSuiteResult = {
	val testInstance = prepareTestInstance(trainTestRatio,fileName)
	val kernel = (new TestingUtils.ScalaObjectsCreator()).rbfKernel(1)
	val optimalHps = obtainOptimalHyperParamsForKernel(testInstance.predictionInput,kernel)
	val optimalKernel = kernel.changeHyperParams(optimalHps.toDenseVector)
	executeTest(testInstance,optimalKernel)(gpMeanPredictor)
  }

  def executeTestForPredictors(trainTestRatio:Double,kernel:KernelFunc,fileName:String):
  	(PredictionTestSuiteResult,PredictionTestSuiteResult,PredictionTestSuiteResult) = {
	val testInstance = prepareTestInstance(trainTestRatio,fileName)
	val optimalHps = obtainOptimalHyperParamsForKernel(testInstance.predictionInput,kernel)
	val optimalKernel = kernel.changeHyperParams(optimalHps.toDenseVector)
	val gpMeanResult = executeTest(testInstance,optimalKernel)(gpMeanPredictor)
	val gpSampleResult = executeTest(testInstance,optimalKernel)(gpSamplePredictor)
	val svmResult = executeTest(testInstance,kernel)(svmPredictor)
	(gpMeanResult,gpSampleResult,svmResult)
  }

  def executeCo2TestForPredictors(trainTestRatio:Double) = {
	val kernel = genericAppContext.getBean("co2Kernel",classOf[Co2Kernel])
	executeTestForPredictors(trainTestRatio, kernel, "co2/maunaLoa2D.txt")
  }

  def executeBostonTestForPredictors(trainTestRatio:Double) = {
	val kernel = genericAppContext.getBean("bostonRbfKernel",classOf[GaussianRbfKernel])
	executeTestForPredictors(trainTestRatio, kernel, "boston.csv")
  }

  def executeTest(testInstance:PredictionTestInstance,kernel:KernelFunc)(executingFunc:testExecutingFunc):PredictionTestSuiteResult = {
	val (predictedValues,ll) = executingFunc(testInstance,kernel)
	val trueValues = testInstance.testTargets
	val mseVal = StatsUtils.mse(predictedValues.toDenseMatrix,trueValues.toDenseMatrix,horSample = false)
	PredictionTestSuiteResult(testTargets = testInstance.testTargets,
	  predictedValues = predictedValues,mse = mseVal,ll = ll)
  }

  def prepareTestInstance(trainTestRatio:Double,fileName:String):PredictionTestInstance = {

	val data:DenseMatrix[Double] = Co2Prediction.loadInput(fileName) //Co2Prediction.loadInput(fileName = fileName)
	val dim = data.cols
	val wholeDsTargets:DenseVector[Double] = data(::,-1)
	val wholeDs:DenseMatrix[Double] = data(::,0 to -2)
	val sampleNum = data.rows
	val trainNum = (trainTestRatio*sampleNum).toInt
	assert(wholeDs.rows == sampleNum)
	assert(wholeDs.cols == dim-1)
	val orderedIds:IndexedSeq[Int] = 0 until data.rows
	val shuffledIds:IndexedSeq[Int] = util.Random.shuffle(orderedIds)
	val shuffledTargets:DenseVector[Double] = DenseVector.tabulate[Double](sampleNum){
	  case id => wholeDsTargets(shuffledIds(id))}
	val shuffledDs:DenseMatrix[Double] = DenseMatrix.zeros[Double](sampleNum,dim-1)
	(0 until sampleNum).foreach{ row =>
	  shuffledDs(row,::) := wholeDs(shuffledIds(row),::)
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
  DenseVector().t

}
