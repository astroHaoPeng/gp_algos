package gp.optimization

import optimization.Optimization
import Optimization._
import breeze.linalg.{DenseVector, DenseMatrix}
import scala.util.Random
import gp.regression.GpPredictor
import breeze.numerics.sqrt
import breeze.optimize.DiffFunction
import utils.MatrixUtils._
import gp.optimization.GPOptimizer.GPOInput
import scala.Some
import org.slf4j.LoggerFactory

/**
 * Created by mjamroz on 14/04/14.
 */
class GPOptimizer(gpPredictor:GpPredictor,noise:Option[Double],gradientOptimizer:GradientBasedOptimizer) {

  val logger = LoggerFactory.getLogger(classOf[GPOptimizer])

  import utils.StatsUtils._
  import utils.KernelRequisites._
  import optimization.Optimization._

  val hyperParams = gpPredictor.kernelFunc.hyperParams

  def minimize(objFunc:objectiveFunction, params: GPOInput): (Array[Double], Double) = {
	val funcToMaximize:objectiveFunction = {point => -objFunc(point)}
	val (optimum,optimumValue) = maximize(funcToMaximize,params)
	(optimum,-optimumValue)
  }

  def maximize(func:objectiveFunction, params: GPOInput): (Array[Double], Double) = {

	val (ranges,m,c,k) = (params.ranges,params.mParam,params.cParam,params.kParam)
	require(c >= 1 && m >= 1, "Params m and c needs to be greater or equal 1")

	val pointGrid:DenseMatrix[Double] = prepareGrid(ranges)
	val evaluatedPointGrid:DenseVector[Double] = evaluateGridPoints(pointGrid,func)
	val hyperParamsForComputation = if (params.optimizeHpOnInitGrid){
	  gpPredictor.obtainOptimalHyperParams(pointGrid,noise,evaluatedPointGrid,true)
	} else {
	  hyperParams
	}
	val (finalPointSet,finalEvaluatedPointSet) = (0 until m).foldLeft((pointGrid,evaluatedPointGrid)){

	  case ((pointSet,evaluatedPointSet),iterNum) =>
		assert(pointSet.rows == evaluatedPointSet.length)
		val (l,alphaVec,_) = gpPredictor.preComputeComponents(pointSet,hyperParamsForComputation,noise,evaluatedPointSet)
		val (observedMean,observedCov) = meanAndVarOfData(pointSet)
		val sampler = new NormalDistributionSampler(GaussianDistribution(mean = observedMean,sigma = observedCov))
	    var (biggestUcbPoint,biggestUcb) = (0 until c).foldLeft[(Option[DenseVector[Double]],Double)](None,Double.MinValue){
		  	case ((currentBiggestUcbPoint,currentBiggestUcb),_) =>
			  val initRandomPoint:DenseVector[Double] = sampler.sample
			  val (ucbValue:Double,pointWithBiggestUCB:DenseVector[Double]) =
				maximizeUCB(l,evaluatedPointSet,pointSet,initRandomPoint,alphaVec,k)
			  if (ucbValue > currentBiggestUcb){(Some(pointWithBiggestUCB),ucbValue)}
			  else {(currentBiggestUcbPoint,currentBiggestUcb)}
		}
		if (biggestUcbPoint.isEmpty){biggestUcbPoint = Some(sampler.sample)}
		logger.debug(s"iteration = ${iterNum}, biggestUcb = ${biggestUcb}, biggestUcbPoint = ${biggestUcbPoint.get}")
		val (newPointSet,newEvaluatedPointSet) = try{
			val evaluatedBiggestUcbPoint = func(biggestUcbPoint.get.toArray)
			(DenseMatrix.vertcat[Double](pointSet,biggestUcbPoint.get.toDenseMatrix),
			  DenseVector.vertcat[Double](evaluatedPointSet,DenseVector(evaluatedBiggestUcbPoint)))
		} catch {
		  case _:Exception => (pointSet,evaluatedPointSet)
		}
		(newPointSet,newEvaluatedPointSet)
	}
	val (maxIndex,maxValue) = (0 until finalEvaluatedPointSet.length).foldLeft((0,Double.MinValue)){
	  case ((biggestIndex,currentBiggestValue),index) =>
		if (finalEvaluatedPointSet(index) > currentBiggestValue){(index,finalEvaluatedPointSet(index))} else {
		  (biggestIndex,currentBiggestValue)
		}
	}
	(finalPointSet(maxIndex,::).toDenseVector.toArray,maxValue)
  }

  private def maximizeUCB(ll:DenseMatrix[Double],targets:DenseVector[Double],pointSet:DenseMatrix[Double],
						  initPoint:DenseVector[Double],alphaVec:DenseVector[Double],kParam:Double):(Double,DenseVector[Double]) = {

	val inversedL:DenseMatrix[Double] = invTriangular(ll,isUpper = false)
	val derAfterFirstArg:kernelDerivative = gpPredictor.kernelFunc.gradient(afterFirstArg = true)
	val derAfterSecondArg:kernelDerivative = gpPredictor.kernelFunc.gradient(afterFirstArg = false)

	val funcForOptimizer:objectiveFunctionWithGradient = {testPoint =>
	  val pointAsDv:DenseVector[Double] = DenseVector(testPoint)
	  val (gaussianPosteriorDistr,vMatrix) = gpPredictor.computePosterior(pointSet,pointAsDv.toDenseMatrix,ll,
		alphaVec,gpPredictor.kernelFunc)
	  assert(vMatrix.cols == 1)
	  require(gaussianPosteriorDistr.mean.length == 1 && gaussianPosteriorDistr.sigma.rows == 1 && gaussianPosteriorDistr.sigma.cols == 1)
	  val ucbObjFunctionValue = gaussianPosteriorDistr.mean(0) + kParam*sqrt(gaussianPosteriorDistr.sigma(0,0))
	  val testTrainDerMtx:DenseMatrix[Double] = testTrainDerMatrix(derAfterFirstArg,pointAsDv,pointSet)
	  val trainTestDerMtx:DenseMatrix[Double] = trainTestDerMatrix(derAfterSecondArg,pointAsDv,pointSet)
	  val derAfterMean = testTrainDerMtx * alphaVec
	  val derAfterVarFirst:DenseVector[Double] = derAfterFirstArg(pointAsDv,pointAsDv)
	  val vAfterXDerMatrix:DenseMatrix[Double] = inversedL * trainTestDerMtx
	  assert(vAfterXDerMatrix.rows == pointSet.rows && vAfterXDerMatrix.cols == pointAsDv.length)
	  val derAfterVar:DenseVector[Double] = derAfterVarFirst - ((vAfterXDerMatrix.t * vMatrix.toDenseVector) :* 2.)
	  val coeff:Double = (kParam/(2*sqrt(gaussianPosteriorDistr.sigma(0,0))))
	  val ucbDer:DenseVector[Double] = derAfterMean + (derAfterVar :* coeff)
	  (ucbObjFunctionValue,ucbDer.toArray)
	}
	val optimalSolution = gradientOptimizer.maximize(funcForOptimizer,initPoint.toArray)
	(funcForOptimizer(optimalSolution)._1,DenseVector(optimalSolution))
  }

  private def buildKernelDerMatrix(kernelDer:kernelDerivative,input1:DenseVector[Double],
								   input2:DenseMatrix[Double],testPointFirstArgToDer:Boolean):DenseMatrix[Double] = {
	val dim = input1.length
	val resultMatrix = if (testPointFirstArgToDer){DenseMatrix.zeros[Double](dim,input2.rows)} else {
	  DenseMatrix.zeros[Double](input2.rows,dim)
	}
	for (index <- (0 until input2.rows)){
	  if (testPointFirstArgToDer){
		val elem = kernelDer(input1,input2(index,::).toDenseVector)
		resultMatrix(::,index) := elem
	  } else {
		val elem = kernelDer(input2(index,::).toDenseVector,input1)
		resultMatrix(index,::) := elem
	  }
	}
	resultMatrix
  }

  private def testTrainDerMatrix(kernelDer:kernelDerivative,
								 testPoint:DenseVector[Double],trainingPoints:DenseMatrix[Double]):DenseMatrix[Double] = {

	buildKernelDerMatrix(kernelDer,testPoint,trainingPoints,true)
  }

  private def trainTestDerMatrix(kernelDer:kernelDerivative,testPoint:DenseVector[Double],
								 trainingPoints:DenseMatrix[Double]):DenseMatrix[Double] = {

	buildKernelDerMatrix(kernelDer,testPoint,trainingPoints,false)
  }

  /*grid(i,::) - i'th d-dimensional point*/
  def evaluateGridPoints(grid:DenseMatrix[Double],func:objectiveFunction):DenseVector[Double] = {
	(0 until grid.rows).foldLeft(DenseVector.zeros[Double](grid.rows)){
	  case (evaluatedPoints,index) => evaluatedPoints.update(index,func(grid(index,::).toDenseVector.toArray)); evaluatedPoints
	}
  }

  def prepareGrid(ranges:IndexedSeq[Range]):DenseMatrix[Double] = {
	val dim = ranges.length
	val resultGrid = DenseMatrix.zeros[Double](3*dim,dim)
	val randGenerator = new Random(System.nanoTime())
	for (i <- (0 until resultGrid.rows)){
		val randomPoint:DenseVector[Double] = ranges.foldLeft((DenseVector.zeros[Double](ranges.length),0)){
		  case ((randVector,index),range) =>
			randVector.update(index,randDouble(randGenerator,range.start,range.end)); (randVector,index+1)
		}._1
	  	resultGrid(i,::) := randomPoint
	}
	resultGrid
  }

  private def randDouble(rand:Random,lower:Double,upper:Double):Double = {
	require(lower < upper)
	lower + (upper - lower)*rand.nextDouble()
  }

}


object GPOptimizer {

  case class GPOInput(ranges:IndexedSeq[Range],mParam:Int,cParam:Int,
					  kParam:Double,optimizeHpOnInitGrid:Boolean=true)

}
