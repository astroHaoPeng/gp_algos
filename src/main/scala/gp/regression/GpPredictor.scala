package gp.regression
import breeze.linalg._
import breeze.optimize.{LBFGS, DiffFunction}
import gp.regression.GpPredictor.{BreezeLBFGSPredictionOptimizer, PredictionTrainingInput, PredictionInput}
import utils.StatsUtils.GaussianDistribution
import utils.KernelRequisites.{KernelFuncHyperParams, KernelFunc}
import org.slf4j.{Logger, LoggerFactory}

/**
 * Created with IntelliJ IDEA.
 * User: mjamroz
 * Date: 30/11/13
 * Time: 16:31
 * To change this template use File | Settings | File Templates.
 */
class GpPredictor(val kernelFunc:KernelFunc) {

  import scala.math._
  import utils.MatrixUtils._
  import GpPredictor._

  type predictOutput = (DenseVector[Double],Double)

  def predict(input:PredictionInput):(GaussianDistribution,Double) = {

	val (trainingData,trainingDataDim,testData,testDataDim) =
	  (input.trainingData,input.trainingData.rows,input.testData,input.testData.rows)
	val newKernelFunc = kernelFunc.changeHyperParams(input.initHyperParams.toDenseVector)
	val (l:DenseMatrix[Double],alphaVec:DenseVector[Double],noiseDiagMtx) =
	  preComputeComponents(trainingData,input.initHyperParams,input.sigmaNoise,input.targets)
	val testTrainCovMatrix:DenseMatrix[Double] = buildKernelMatrix(newKernelFunc,testData,trainingData)
	assert(testTrainCovMatrix.rows == testDataDim && testTrainCovMatrix.cols == trainingDataDim)
	val fMean:DenseVector[Double] = (testTrainCovMatrix * alphaVec).toDenseVector
	val vMatrix:DenseMatrix[Double] = forwardSolve(L = l,b = testTrainCovMatrix.t)
	assert(vMatrix.rows == trainingDataDim && vMatrix.cols == testDataDim)
	val fVariance:DenseMatrix[Double] = buildKernelMatrix(newKernelFunc,testData) - (vMatrix.t * vMatrix)
	val fVarianceWithNoise:DenseMatrix[Double] = if (input.sigmaNoise.isDefined){
	  fVariance + noiseDiagMtx.get
	} else {fVariance}
	assert(fMean.length == testDataDim && fVariance.rows == testDataDim && fVariance.cols == testDataDim)
	val logLikelihoodVal:Double = logLikelihood(alphaVec,l,input.targets)
	(GaussianDistribution(mean = fMean,sigma = fVarianceWithNoise),logLikelihoodVal)
  }

  def computePosterior(trainingData:DenseMatrix[Double],testData:DenseMatrix[Double],l:DenseMatrix[Double],
					   alphaVec:DenseVector[Double]):(GaussianDistribution,DenseMatrix[Double]) = {
	computePosterior(trainingData,testData,l,alphaVec,kernelFunc)
  }

  def computePosterior(trainingData:DenseMatrix[Double],testData:DenseMatrix[Double],l:DenseMatrix[Double],
					   alphaVec:DenseVector[Double],kernelFunc:KernelFunc):(GaussianDistribution,DenseMatrix[Double]) = {

	val testTrainCovMatrix:DenseMatrix[Double] = buildKernelMatrix(kernelFunc,testData,trainingData)
	val fMean:DenseVector[Double] = (testTrainCovMatrix * alphaVec).toDenseVector
	val vMatrix:DenseMatrix[Double] = forwardSolve(L = l,b = testTrainCovMatrix.t)
	val fVariance:DenseMatrix[Double] = buildKernelMatrix(kernelFunc,testData) - (vMatrix.t * vMatrix)
	(GaussianDistribution(mean = fMean,sigma = fVariance), vMatrix)
  }

  def logLikelihoodWithDerivatives(input:PredictionTrainingInput,hyperParams:KernelFuncHyperParams,
								   optimizedParamsNum:Int):
  		(Double,DenseVector[Double]) = {
  	val (l:DenseMatrix[Double],alphaVec:DenseVector[Double],_) =
	  preComputeComponents(input.trainingData,hyperParams,input.sigmaNoise,input.targets)
	val ll = logLikelihood(alphaVec,l,input.targets)
	val lInversed:DenseMatrix[Double] = invTriangular(l,isUpper = false)
	val inversedK:DenseMatrix[Double] = lInversed.t * lInversed
	val newKernelFunc:KernelFunc = kernelFunc.changeHyperParams(hyperParams.toDenseVector)
	val alphaSq:DenseMatrix[Double] = alphaVec * alphaVec.t
	val gradient = (0 until optimizedParamsNum).foldLeft(DenseVector.zeros[Double](optimizedParamsNum)){
	  case (gradient,index) =>
		val func:(DenseVector[Double],DenseVector[Double],Boolean) => Double = {(vec1,vec2,sameIndex) =>
		  newKernelFunc.derAfterHyperParam(index+1)(vec1,vec2,sameIndex)
		}
		val derAfterKernelHyperParams:DenseMatrix[Double] = buildMatrixWithFunc(input.trainingData)(func)
		val logLikelihoodDerAfterParam:Double = 0.5*trace((alphaSq - inversedK) * derAfterKernelHyperParams)
		gradient.update(index,logLikelihoodDerAfterParam); gradient
	}
	(ll,gradient)
  }

  def predictWithParamsOptimization(input:PredictionInput,optimizeNoise:Boolean):(GaussianDistribution,Double) = {
	val optimizer = new BreezeLBFGSPredictionOptimizer(this,optimizeNoise)
	val optimizedParams:KernelFuncHyperParams = optimizer.optimizerHyperParams(input)
	predict(input.copy(initHyperParams = optimizedParams))
  }

  def preComputeComponents(trainingData:DenseMatrix[Double],
						   sigmaNoise:Option[Double],targets:DenseVector[Double]):
  afterLearningComponents = {

	preComputeComponents(trainingData,kernelFunc.hyperParams,sigmaNoise,targets)
  }


  def preComputeComponents(trainingData:DenseMatrix[Double],hyperParams:KernelFuncHyperParams,
								   sigmaNoise:Option[Double],targets:DenseVector[Double]):
  	afterLearningComponents = {

	require(trainingData.rows == targets.length,
	  "Number of objects in training data matrix should be equal to targets vector length")
	val trainingDataDim = trainingData.rows
	val newKernelFunc = kernelFunc.changeHyperParams(hyperParams.toDenseVector)
	val kernelMatrixWithoutNoise:DenseMatrix[Double] = buildKernelMatrix(newKernelFunc,trainingData)
	val (kernelMatrixAfterOptionalNoiseAddition:DenseMatrix[Double],
		noiseDiagMtx:Option[DenseMatrix[Double]]) = sigmaNoise match {
	  case Some(sigmaNoise_) =>
		val noiseDiagMtx:DenseMatrix[Double] = DenseMatrix.eye[Double](trainingDataDim) :* sigmaNoise_
		(kernelMatrixWithoutNoise + noiseDiagMtx,Some(noiseDiagMtx))
	  case None => (kernelMatrixWithoutNoise,None)
	}
	val L = cholesky(kernelMatrixAfterOptionalNoiseAddition)
	val temp:DenseVector[Double] = forwardSolve(L = L,b = targets)
	val alphaVec:DenseVector[Double] = backSolve(R = L.t,b = temp)
	(L,alphaVec,noiseDiagMtx)
  }

  private def logLikelihood(alphaVector:DenseVector[Double],L:DenseMatrix[Double],targets:DenseVector[Double]):Double = {
	val n = L.rows
	val a1 = -0.5*(targets dot alphaVector)
	val a2 = 0.to(n-1).foldLeft[Double](0.){case (sum,indx) => sum + log(L(indx,indx))}
	a1 - a2 - 0.5*n*log(2*Pi)
  }

}

object GpPredictor {

  /*_1 - L - cholesky decomposition of kernel matrix, _2 - alphaVector = L.t \ (L \ targets),
  _3 - noise diag matrix = noise * I*/
  type afterLearningComponents = (DenseMatrix[Double],DenseVector[Double],Option[DenseMatrix[Double]])

  val apacheLogger:Logger = LoggerFactory.getLogger(classOf[GpPredictor])

  trait PredictionHyperParamsOptimizer {

	def optimizerHyperParams(predictionInput:PredictionInput):KernelFuncHyperParams
	
  }

  /*Noise can also be incorporated into kernel function, then sigmaNoise should be set to None*/
  case class PredictionInput(trainingData:DenseMatrix[Double],testData:DenseMatrix[Double],
							 sigmaNoise:Option[Double],targets:DenseVector[Double],
							 initHyperParams:KernelFuncHyperParams) {

	def toPredictionTrainingInput:PredictionTrainingInput = {
		PredictionTrainingInput(trainingData = trainingData,sigmaNoise = sigmaNoise,
		  targets = targets,initHyperParams = initHyperParams)
	}
  }

  case class PredictionTrainingInput (trainingData:DenseMatrix[Double],sigmaNoise:Option[Double],
									  targets:DenseVector[Double],initHyperParams:KernelFuncHyperParams)
  
  //TODO - unify params optimization in classification and prediction problems
  class BreezeLBFGSPredictionOptimizer (gpPredictor:GpPredictor,optimizeNoise:Boolean)
	extends PredictionHyperParamsOptimizer{

	def optimizerHyperParams(predictionInput: PredictionInput): KernelFuncHyperParams = {

	  /*diffFunction will be minimized so it needs to be equal to -logLikelihood*/
	  val diffFunction = new DiffFunction[DenseVector[Double]] {

		def calculate(hyperParams: DenseVector[Double]): (Double, DenseVector[Double]) = {

		  val hyperParamsVec:KernelFuncHyperParams = predictionInput.initHyperParams.fromDenseVector(hyperParams)
		  val (logLikelihood,derivatives) = gpPredictor.logLikelihoodWithDerivatives(
			predictionInput.toPredictionTrainingInput,hyperParamsVec,
		  	hyperParams.length)
		  assert(hyperParams.length == derivatives.length)
		  apacheLogger.info(s"Current solution is = ${hyperParams}, objective function value = ${-logLikelihood}")
		  (-logLikelihood,derivatives :* (-1.))
		}
	  }

	  val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 30,m = 3)
	  val initParams = if (optimizeNoise){predictionInput.initHyperParams.toDenseVector} else {
		predictionInput.initHyperParams.toDenseVector(0 to -2)
	  }
	  val optimizedParams = lbfgs.minimize(diffFunction,initParams)
	  predictionInput.initHyperParams.fromDenseVector(optimizedParams)
	}
  }

}




