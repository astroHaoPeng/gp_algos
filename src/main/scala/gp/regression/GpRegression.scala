package gp.regression
import breeze.linalg._
import org.springframework.core.io.ClassPathResource
import breeze.io.CSVReader
import breeze.optimize.{StochasticDiffFunction, LBFGS, DiffFunction}
import breeze.optimize.StochasticGradientDescent.SimpleSGD
import gp.regression.GpRegression.{PredictionTrainingInput, PredictionInput}
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
	/*val (trainingData,trainingDataDim,testData,testDataDim) =
	  (input.trainingData,input.trainingData.rows,input.testData,input.testData.rows)
	val newKernelFunc = kernelFunc.changeHyperParams(input.initHyperParams.toDenseVector)
	val kernelMatrixWithoutNoise = buildKernelMatrix(newKernelFunc,trainingData)
	val (kernelMatrixAfterOptionalNoiseAddition:DenseMatrix[Double],
		noiseDiagMtx:Option[DenseMatrix[Double]]) = input.sigmaNoise match {
	  case Some(sigmaNoise_) =>
		val noiseDiagMtx:DenseMatrix[Double] = DenseMatrix.eye[Double](trainingDataDim) :* sigmaNoise_
		(kernelMatrixWithoutNoise + noiseDiagMtx,Some(noiseDiagMtx))
	  case None => (kernelMatrixWithoutNoise,None)
	}
	//val testTrainCovMatrix:DenseMatrix[Double] = testTrainKernelMatrix(testData,trainingData,kernelFunc)
	val testTrainCovMatrix:DenseMatrix[Double] = buildKernelMatrix(newKernelFunc,testData,trainingData)
	assert(testTrainCovMatrix.rows == testDataDim && testTrainCovMatrix.cols == trainingDataDim)
	val L = cholesky(kernelMatrixAfterOptionalNoiseAddition)
	val temp:DenseVector[Double] = forwardSolve(L = L,b = input.targets)
	val alphaVec:DenseVector[Double] = backSolve(R = L.t,b = temp) */
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

  def logLikelihoodWithDerivatives(input:PredictionTrainingInput,hyperParams:DenseVector[Double]):
  		(Double,DenseVector[Double]) = {
  	val (l:DenseMatrix[Double],alphaVec:DenseVector[Double],_) =
	  preComputeComponents(input.trainingData,input.initHyperParams,input.sigmaNoise,input.targets)
	val ll = logLikelihood(alphaVec,l,input.targets)
	val lInversed:DenseMatrix[Double] = invTriangular(l,isUpper = false)
	val inversedK:DenseMatrix[Double] = lInversed.t * lInversed
	val newKernelFunc = kernelFunc.changeHyperParams(hyperParams)
	val alphaSq:DenseMatrix[Double] = alphaVec * alphaVec.t
	val gradient = (0 until newKernelFunc.hyperParametersNum).foldLeft(DenseVector.zeros[Double](newKernelFunc.hyperParametersNum)){
	  case (gradient,index) =>
		val func:(DenseVector[Double],DenseVector[Double]) => Double = {(vec1,vec2) =>
		  kernelFunc.derAfterHyperParam(index+1)(vec1,vec2)
		}
		val derAfterKernelHyperParams:DenseMatrix[Double] = buildKernelMatrix(input.trainingData)(func)
		val logLikelihoodDerAfterParam:Double = 0.5*trace((alphaSq - inversedK) * derAfterKernelHyperParams)
		gradient.update(index,logLikelihoodDerAfterParam); gradient
	}
	(ll,gradient)
  }

  private def preComputeComponents(trainingData:DenseMatrix[Double],initHyperParams:KernelFuncHyperParams,
								   sigmaNoise:Option[Double],targets:DenseVector[Double]):
  	(DenseMatrix[Double],DenseVector[Double],Option[DenseMatrix[Double]]) = {

	val trainingDataDim = trainingData.rows
	val newKernelFunc = kernelFunc.changeHyperParams(initHyperParams.toDenseVector)
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


  /*
  def predictWithOptimization(example:DenseVector[Double],training:DenseMatrix[Double],targets:DenseVector[Double],
							  params:GaussianRbfParams=defaultRbfParams)
  	:(Double,Double) = {
	val prediction = predictWithOptimization_(example.toDenseMatrix,training,targets,params)
	(prediction._1(0),prediction._2)
  }

  def predictWithOptimization_(testData:DenseMatrix[Double],training:DenseMatrix[Double],targets:DenseVector[Double],
							   kernelParams:GaussianRbfParams=defaultRbfParams)
  :predictOutput = {
	val kernelFunc = GaussianRbfKernel(kernelParams)
	val kernelMatrix = buildKernelMatrix(kernelFunc,training,beta = beta)
	val L = cholesky(kernelMatrix)
	val alphaVector = (L.t \ (L \ targets))
	val rbfParams:GaussianRbfParams = optimizeParams(alphaVector,L,training,targets)
	predict(testData,training,targets,rbfParams)
  }  */

  private def logLikelihood(alphaVector:DenseVector[Double],L:DenseMatrix[Double],targets:DenseVector[Double]):Double = {
	val n = L.rows
	val a1 = -0.5*(targets dot alphaVector)
	val a2 = 0.to(n-1).foldLeft[Double](0.){case (sum,indx) => sum + log(L(indx,indx))}
	a1 - a2 - 0.5*n*log(2*Pi)
  }

  /*
  private def optimizeParams(alphaVector:DenseVector[Double],L:DenseMatrix[Double],training:DenseMatrix[Double],
							  targets:DenseVector[Double]):GaussianRbfParams = {
	val n = L.rows
	type kernelFunByIndex = (Int,Int) => Double
	val applyDerivative: kernelFunByIndex => DenseMatrix[Double] = {kerFun =>
	  val result = DenseMatrix.zeros[Double](n,n)
	  for (i <- 0.to(n-1)){
		for (j <- 0.to(n-1)){
		  result.update(i,j,kerFun(i,j))
		}
	  }
	  result
	}

	val objFunction:StochasticDiffFunction[DenseVector[Double]] = new StochasticDiffFunction[DenseVector[Double]] {
	  def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
		val (a,g,b) = (x(0),x(1),x(2))
		val kernelFunc = GaussianRbfKernel(GaussianRbfParams(alpha = a,gamma = g))
		val kernelMatrix = buildKernelMatrix(kernelFunc,training)
		val L = cholesky(kernelMatrix)
		val inversedK:DenseMatrix[Double] = (L.t \ (L \ DenseMatrix.eye[Double](n)))
		val alphaVector = inversedK * targets
		val objValue = logLikelihood(alphaVector,L,targets)
		val alphaSq:DenseMatrix[Double] = alphaVector * alphaVector.t
		assert(alphaSq.rows == n && alphaSq.cols == n)
		val mlAfterAlphaDer:kernelFunByIndex = {(i,j) =>
		  val diff = (training(i,::) - training(j,::)).toDenseVector
		  exp(-0.5*g*(diff dot diff))
		}
		val mlAfterGammaDer:kernelFunByIndex = {(i,j) =>
		  val diff = (training(i,::) - training(j,::)).toDenseVector; val prod = diff dot diff
		  a*exp(-0.5*g*prod)*(-0.5*prod)
		}
		val mlAfterBetaDer:kernelFunByIndex = {(i,j) =>
		  i == j match {
			case true => -1./(b*b)
			case false => 0
		  }
		}
		val m:DenseMatrix[Double] = (alphaSq-inversedK)*applyDerivative(mlAfterAlphaDer)
		val m1:DenseMatrix[Double] = (alphaSq-inversedK)*applyDerivative(mlAfterGammaDer)
		val m2:DenseMatrix[Double] = (alphaSq-inversedK)*applyDerivative(mlAfterBetaDer)
		(-objValue,DenseVector(trace(m),trace(m1),trace(m2)) * -0.5)
	  }
	}
//	val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 100,m = 3)
//	val res = lbfgs.minimize(objFunction,DenseVector(alpha,gamma,beta))
	val sg = new SimpleSGD[DenseVector[Double]](maxIter = 3000)
	val res = sg.minimize(objFunction,DenseVector(alpha,gamma,beta))
	GaussianRbfParams(alpha = res(0),gamma = res(1))
  } */
}

object GpRegression {

  val apacheLogger:Logger = LoggerFactory.getLogger(classOf[GpRegression])
  
  trait PredictionHyperParamsOptimizer {
	
	def optimizerHyperParams(predictionInput:PredictionInput):KernelFuncHyperParams
	
  }

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
  /*class BreezeLBFGSPredictionOptimizer(gpPredictor:GpRegression) extends PredictionHyperParamsOptimizer {
	
	def optimizerHyperParams(predictionInput: PredictionInput): KernelFuncHyperParams = {
	  val (trainData,targets) = (predictionInput.trainingData,predictionInput.targets)

	  /*diffFunction will be minimized so it needs to be equal to -logLikelihood*/
	  val diffFunction = new DiffFunction[DenseVector[Double]] {

		def calculate(hyperParams: DenseVector[Double]): (Double, DenseVector[Double]) = {

		  val (_,logLikelihood) =
		  val (logLikelihood,derivatives) = marginalLikelihoodEvaluator.logLikelihood(trainData,targets,hyperParams)
		  assert(hyperParams.length == derivatives.length)
		  apacheLogger.info(s"Current solution is = ${hyperParams}, objective function value = ${-logLikelihood}")
		  (-logLikelihood,derivatives :* (-1.))
		}
	  }

	  val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 10,m = 3)
	  val optimizedParams = lbfgs.minimize(diffFunction,optimizationInput.initHyperParams.toDenseVector)
	  optimizationInput.initHyperParams.fromDenseVector(optimizedParams)  
	}
  } */

}




