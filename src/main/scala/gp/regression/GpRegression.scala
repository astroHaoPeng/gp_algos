package gp.regression
import breeze.linalg._
import org.springframework.core.io.ClassPathResource
import breeze.io.CSVReader
import breeze.optimize.{StochasticDiffFunction, LBFGS, DiffFunction}
import breeze.optimize.StochasticGradientDescent.SimpleSGD
import gp.regression.GpRegression.PredictionInput
import utils.StatsUtils.GaussianDistribution
import utils.KernelRequisites.{KernelFuncHyperParams, GaussianRbfKernel, GaussianRbfParams, KernelFunc}

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
	val kernelMatrixWithoutNoise = buildKernelMatrix(kernelFunc,trainingData)
	val (kernelMatrixAfterOptionalNoiseAddition:DenseMatrix[Double],
		noiseDiagMtx:Option[DenseMatrix[Double]]) = input.sigmaNoise match {
	  case Some(sigmaNoise_) =>
		val noiseDiagMtx:DenseMatrix[Double] = DenseMatrix.eye[Double](trainingDataDim) :* sigmaNoise_
		(kernelMatrixWithoutNoise + noiseDiagMtx,Some(noiseDiagMtx))
	  case None => (kernelMatrixWithoutNoise,None)
	}
	//val testTrainCovMatrix:DenseMatrix[Double] = testTrainKernelMatrix(testData,trainingData,kernelFunc)
	val testTrainCovMatrix:DenseMatrix[Double] = buildKernelMatrix(kernelFunc,testData,trainingData)
	assert(testTrainCovMatrix.rows == testDataDim && testTrainCovMatrix.cols == trainingDataDim)
	val L = cholesky(kernelMatrixAfterOptionalNoiseAddition)
	val temp:DenseVector[Double] = forwardSolve(L = L,b = input.targets)
	val alphaVec:DenseVector[Double] = backSolve(R = L.t,b = temp)
	val fMean:DenseVector[Double] = (testTrainCovMatrix * alphaVec).toDenseVector
	val vMatrix:DenseMatrix[Double] = forwardSolve(L = L,b = testTrainCovMatrix.t)
	assert(vMatrix.rows == trainingDataDim && vMatrix.cols == testDataDim)
	val fVariance:DenseMatrix[Double] = buildKernelMatrix(kernelFunc,testData) - (vMatrix.t * vMatrix)
	val fVarianceWithNoise:DenseMatrix[Double] = if (input.sigmaNoise.isDefined){
	  fVariance + noiseDiagMtx.get
	} else {fVariance}
	assert(fMean.length == testDataDim && fVariance.rows == testDataDim && fVariance.cols == testDataDim)
	val logLikelihoodVal:Double = logLikelihood(alphaVec,L,input.targets)
	(GaussianDistribution(mean = fMean,sigma = fVarianceWithNoise),logLikelihoodVal)
  }

  /*
  def predict(example:DenseVector[Double],training:DenseMatrix[Double],targets:DenseVector[Double])
  	:(Double,Double) = {
  	val prediction = predict(example.toDenseMatrix,training,targets)
	(prediction._1(0),prediction._2)
  }

  def predict(testData:DenseMatrix[Double],training:DenseMatrix[Double],targets:DenseVector[Double],
			  kernelParams:GaussianRbfParams=defaultRbfParams):predictOutput = {

	val kernelFunc = GaussianRbfKernel(kernelParams)
	val kernelMatrix = buildKernelMatrix(kernelFunc,training,beta = beta)
	val testTrainCov = testTrainKernelMatrix(testData,training,kernelFunc)
	val L = cholesky(kernelMatrix)
	val alphaVector = (L.t \ (L \ targets))
	((testTrainCov * alphaVector).toDenseVector,logLikelihood(alphaVector,L,targets))
  }
    */

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

  case class PredictionInput(trainingData:DenseMatrix[Double],testData:DenseMatrix[Double],
							 sigmaNoise:Option[Double],targets:DenseVector[Double],
									  initHyperParams:KernelFuncHyperParams)

}




