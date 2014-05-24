package gp.regression
import breeze.linalg._
import utils.StatsUtils.GaussianDistribution
import utils.KernelRequisites.{KernelFuncHyperParams, KernelFunc}
import org.slf4j.{Logger, LoggerFactory}
import optimization.Optimization.BreezeLbfgsOptimizer

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
  import optimization._

  type predictOutput = (DenseVector[Double],Double)

  def predict(input:PredictionInput,hyperParams:KernelFuncHyperParams=kernelFunc.hyperParams):(GaussianDistribution,Double) = {

	val (trainingData,trainingDataDim,testData,testDataDim) =
	  (input.trainingData,input.trainingData.rows,input.testData,input.testData.rows)
	val newKernelFunc = kernelFunc.changeHyperParams(hyperParams.toDenseVector)
	val (l:DenseMatrix[Double],alphaVec:DenseVector[Double],noiseDiagMtx) =
	  preComputeComponents(trainingData,hyperParams,input.sigmaNoise,input.targets)
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
	val gradientVec = (0 until optimizedParamsNum).foldLeft(DenseVector.zeros[Double](optimizedParamsNum)){
	  case (gradient,index) =>
		val func:(DenseVector[Double],DenseVector[Double],Boolean) => Double = {(vec1,vec2,sameIndex) =>
		  newKernelFunc.derAfterHyperParam(index+1)(vec1,vec2,sameIndex)
		}
		val derAfterKernelHyperParams:DenseMatrix[Double] = buildMatrixWithFunc(input.trainingData)(func)
		val logLikelihoodDerAfterParam:Double = 0.5*trace((alphaSq - inversedK) * derAfterKernelHyperParams)
		gradient.update(index,logLikelihoodDerAfterParam); gradient
	}
	(ll,gradientVec)
  }

  def predictWithParamsOptimization(input:PredictionInput,optimizeNoise:Boolean):(GaussianDistribution,Double,KernelFuncHyperParams) = {
	val optimalHyperParams:KernelFuncHyperParams = obtainOptimalHyperParams(trainingData = input.trainingData,
	targets = input.targets,sigmaNoise = input.sigmaNoise,optimizeNoise = optimizeNoise)
	val predictedValWithLL = predict(input,hyperParams = optimalHyperParams)
	(predictedValWithLL._1,predictedValWithLL._2,optimalHyperParams)
  }

  def preComputeComponents(trainingData:DenseMatrix[Double],
						   sigmaNoise:Option[Double],targets:DenseVector[Double]):
  afterLearningComponents = {

	preComputeComponents(trainingData,kernelFunc.hyperParams,sigmaNoise,targets)
  }

  def preComputeComponentsWithHpOptimization(trainingData:DenseMatrix[Double],sigmaNoise:Option[Double],
											 targets:DenseVector[Double]):
  			(afterLearningComponents,KernelFuncHyperParams) = {
	val optimalHyperParams:KernelFuncHyperParams = obtainOptimalHyperParams(trainingData,sigmaNoise,targets,false)
	(preComputeComponents(trainingData,optimalHyperParams,sigmaNoise,targets),optimalHyperParams)
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

  private def obtainOptimalHyperParams(trainingData:DenseMatrix[Double],sigmaNoise:Option[Double],
									   targets:DenseVector[Double],optimizeNoise:Boolean):KernelFuncHyperParams = {

	val breezeOptimizer = new BreezeLbfgsOptimizer(maxIter = 20)
	val initPoint:DenseVector[Double] = if (optimizeNoise){kernelFunc.hyperParams.toDenseVector} else{
	  kernelFunc.hyperParams.toDenseVector(0 to -2)
	}
	val llObjFunction:Optimization.objectiveFunctionWithGradient = { currentParams =>
	  	val hyperParams:KernelFuncHyperParams = kernelFunc.hyperParams.fromDenseVector(DenseVector(currentParams))
		val ptInput = PredictionTrainingInput(trainingData = trainingData,targets = targets,
			sigmaNoise = sigmaNoise)
	  	val (value,gradient) = logLikelihoodWithDerivatives(ptInput,hyperParams,currentParams.length)
	  	(value,gradient.toArray)
	}
	val optimalHyperParams = breezeOptimizer.maximize(llObjFunction,initPoint.toArray)
	kernelFunc.hyperParams.fromDenseVector(DenseVector(optimalHyperParams))
  }

}

object GpPredictor {

  /*_1 - L - cholesky decomposition of kernel matrix, _2 - alphaVector = L.t \ (L \ targets),
  _3 - noise diag matrix = noise * I*/
  type afterLearningComponents = (DenseMatrix[Double],DenseVector[Double],Option[DenseMatrix[Double]])

  val apacheLogger:Logger = LoggerFactory.getLogger(classOf[GpPredictor])

  /*Noise can also be incorporated into kernel function, then sigmaNoise should be set to None*/
  case class PredictionInput(trainingData:DenseMatrix[Double],testData:DenseMatrix[Double],
							 sigmaNoise:Option[Double],targets:DenseVector[Double]) {

	def toPredictionTrainingInput:PredictionTrainingInput = {
		PredictionTrainingInput(trainingData = trainingData,sigmaNoise = sigmaNoise,
		  targets = targets)
	}
  }

  case class PredictionTrainingInput (trainingData:DenseMatrix[Double],sigmaNoise:Option[Double],
									  targets:DenseVector[Double])
  


}




