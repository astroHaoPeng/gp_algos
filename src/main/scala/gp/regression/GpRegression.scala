package gp.regression
import breeze.linalg._
import org.springframework.core.io.ClassPathResource
import breeze.io.CSVReader
import breeze.optimize.{StochasticDiffFunction, LBFGS, DiffFunction}
import breeze.optimize.StochasticGradientDescent.SimpleSGD
import gp.regression.GpRegression.{BreezeLBFGSPredictionOptimizer, PredictionTrainingInput, PredictionInput}
import utils.StatsUtils.GaussianDistribution
import utils.KernelRequisites.{KernelFuncHyperParams, GaussianRbfKernel, GaussianRbfParams, KernelFunc}
import org.slf4j.{Logger, LoggerFactory}

/**
 * Created with IntelliJ IDEA.
 * User: mjamroz
 * Date: 30/11/13
 * Time: 16:31
 * To change this template use File | Settings | File Templates.
 */
class GpRegression(kernelFunc:KernelFunc) {

  import scala.math._
  import utils.MatrixUtils._

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

  def logLikelihoodWithDerivatives(input:PredictionTrainingInput,hyperParams:KernelFuncHyperParams):
  		(Double,DenseVector[Double]) = {
  	val (l:DenseMatrix[Double],alphaVec:DenseVector[Double],_) =
	  preComputeComponents(input.trainingData,hyperParams,input.sigmaNoise,input.targets)
	val ll = logLikelihood(alphaVec,l,input.targets)
	val lInversed:DenseMatrix[Double] = invTriangular(l,isUpper = false)
	val inversedK:DenseMatrix[Double] = lInversed.t * lInversed
	val newKernelFunc = kernelFunc.changeHyperParams(hyperParams.toDenseVector)
	val alphaSq:DenseMatrix[Double] = alphaVec * alphaVec.t
	val gradient = (0 until newKernelFunc.hyperParametersNum).foldLeft(DenseVector.zeros[Double](newKernelFunc.hyperParametersNum)){
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

  def predictWithParamsOptimization(input:PredictionInput):(GaussianDistribution,Double) = {
	val optimizer = new BreezeLBFGSPredictionOptimizer(this)
	val optimizedParams:KernelFuncHyperParams = optimizer.optimizerHyperParams(input)
	predict(input.copy(initHyperParams = optimizedParams))
  }

  private def preComputeComponents(trainingData:DenseMatrix[Double],hyperParams:KernelFuncHyperParams,
								   sigmaNoise:Option[Double],targets:DenseVector[Double]):
  	(DenseMatrix[Double],DenseVector[Double],Option[DenseMatrix[Double]]) = {

	val trainingDataDim = trainingData.rows
	val newKernelFunc = kernelFunc.changeHyperParams(hyperParams.toDenseVector)
	val kernelMatrixWithoutNoise = buildKernelMatrix(newKernelFunc,trainingData)
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

object GpRegression {

  val apacheLogger:Logger = LoggerFactory.getLogger(classOf[GpRegression])
  
  trait PredictionHyperParamsOptimizer {
	
	def optimizerHyperParams(predictionInput:PredictionInput):KernelFuncHyperParams
	
  }

  /*Noise can also be incorporated into kernel function, then sigmaNoise should be set to None*/
  case class PredictionInput(trainingData:DenseMatrix[Double],testData:DenseMatrix[Double],
							 sigmaNoise:Option[Double],targets:DenseVector[Double],
							 initHyperParams:KernelFuncHyperParams){

	def toPredictionTrainingInput:PredictionTrainingInput = {
		PredictionTrainingInput(trainingData = trainingData,sigmaNoise = sigmaNoise,
		  targets = targets,initHyperParams = initHyperParams)
	}
  }

  case class PredictionTrainingInput(trainingData:DenseMatrix[Double],sigmaNoise:Option[Double],
									  targets:DenseVector[Double],initHyperParams:KernelFuncHyperParams)
  
  //TODO - unify params optimization in classification and prediction problems
  class BreezeLBFGSPredictionOptimizer(gpPredictor:GpRegression) extends PredictionHyperParamsOptimizer {
	
	def optimizerHyperParams(predictionInput: PredictionInput): KernelFuncHyperParams = {

	  /*diffFunction will be minimized so it needs to be equal to -logLikelihood*/
	  val diffFunction = new DiffFunction[DenseVector[Double]] {

		def calculate(hyperParams: DenseVector[Double]): (Double, DenseVector[Double]) = {

		  val (logLikelihood,derivatives) = gpPredictor.logLikelihoodWithDerivatives(
			predictionInput.toPredictionTrainingInput,predictionInput.initHyperParams.fromDenseVector(hyperParams))
		  assert(hyperParams.length == derivatives.length)
		  apacheLogger.info(s"Current solution is = ${hyperParams}, objective function value = ${-logLikelihood}")
		  (-logLikelihood,derivatives :* (-1.))
		}
	  }

	  val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 30,m = 3)
	  val optimizedParams = lbfgs.minimize(diffFunction,predictionInput.initHyperParams.toDenseVector)
	  predictionInput.initHyperParams.fromDenseVector(optimizedParams)
	}
  }

}




