package gp.optimization

import Optimization._
import GPOptimizer._
import breeze.linalg.{max, DenseVector, DenseMatrix}
import scala.util.Random
import gp.regression.GpPredictor
import breeze.numerics.sqrt
import breeze.optimize.{LBFGS, DiffFunction}
import utils.MatrixUtils._
import gp.optimization.GPOptimizer.GPOInput
import scala.Some
import org.slf4j.LoggerFactory

/**
 * Created by mjamroz on 14/04/14.
 */
class GPOptimizer(gpPredictor:GpPredictor,noise:Option[Double]) extends Optimizer[(Array[Double],Double),GPOInput]{

  val logger = LoggerFactory.getLogger(classOf[GPOptimizer])

  import utils.StatsUtils._
  import utils.KernelRequisites._
  
  type objectiveFunction = (Array[Double] => Double)

  val hyperParams = gpPredictor.kernelFunc.hyperParams

  def minimize(objFunc:objectiveFunction, params: GPOInput): (Array[Double], Double) = ???

  def maximize(func:objectiveFunction, params: GPOInput): (Array[Double], Double) = {

	val (ranges,m,c,k) = (params.ranges,params.mParam,params.cParam,params.kParam)
	require(c >= 1 && m >= 1, "Params m and c needs to be greater or equal 1")

	val pointGrid:DenseMatrix[Double] = prepareGrid(ranges)
	val evaluatedPointGrid:DenseVector[Double] = evaluateGridPoints(pointGrid,func)

	val (finalPointSet,finalEvaluatedPointSet) = (0 until m).foldLeft((pointGrid,evaluatedPointGrid)){

	  case ((pointSet,evaluatedPointSet),iterNum) =>
		val (l,alphaVec,_) = gpPredictor.preComputeComponents(pointSet,hyperParams,noise,evaluatedPointSet)
		val (observedMean,observedCov) = meanAndVarOfData(pointSet)
		val sampler = new NormalDistributionSampler(GaussianDistribution(mean = observedMean,sigma = observedCov))
	    val (biggestUcbPoint,biggestUcb) = (0 until c).foldLeft[(Option[DenseVector[Double]],Double)](None,Double.MinValue){
		  	case ((currentBiggestUcbPoint,currentBiggestUcb),_) =>
			  val initRandomPoint:DenseVector[Double] = sampler.sample
			  val (ucbValue:Double,pointWithBiggestUCB:DenseVector[Double]) =
				maximizeUCB(l,evaluatedPointSet,pointSet,initRandomPoint,alphaVec,k)
			  if (ucbValue > currentBiggestUcb){(Some(pointWithBiggestUCB),ucbValue)}
			  else {(currentBiggestUcbPoint,currentBiggestUcb)}
		}
		logger.info(s"iteration = ${iterNum}, biggestUcb = ${biggestUcb}, biggestUcbPoint = ${biggestUcbPoint.get}")
		(DenseMatrix.vertcat[Double](pointSet,biggestUcbPoint.get.toDenseMatrix),
		  DenseVector.vertcat[Double](evaluatedPointGrid,DenseVector(func(biggestUcbPoint.get.data))))
	}
	val (maxIndex,maxValue) = (0 until evaluatedPointGrid.length).foldLeft((0,Double.MinValue)){
	  case ((biggestIndex,currentBiggestValue),index) =>
		if (finalEvaluatedPointSet(index) > currentBiggestValue){(index,currentBiggestValue)} else {
		  (biggestIndex,currentBiggestValue)
		}
	}
	(finalPointSet(maxIndex,::).data,maxValue)
  }

  private def maximizeUCB(ll:DenseMatrix[Double],targets:DenseVector[Double],pointSet:DenseMatrix[Double],
						  initPoint:DenseVector[Double],alphaVec:DenseVector[Double],kParam:Double):(Double,DenseVector[Double]) = {

	val inversedL:DenseMatrix[Double] = invTriangular(ll,isUpper = false)
	val derAfterFirstArg:kernelDerivative = gpPredictor.kernelFunc.gradient(afterFirstArg = true)
	val diffFunction = new DiffFunction[DenseVector[Double]] {
	  override def calculate(testPoint: DenseVector[Double]): (Double, DenseVector[Double]) = {
		val (gaussianPosteriorDistr,vMatrix) = gpPredictor.computePosterior(pointSet,testPoint.toDenseMatrix,ll,
		  alphaVec,gpPredictor.kernelFunc)
		assert(vMatrix.cols == 1)
		require(gaussianPosteriorDistr.mean.length == 1 && gaussianPosteriorDistr.sigma.rows == 1 && gaussianPosteriorDistr.sigma.cols == 1)
		val ucbObjFunctionValue = gaussianPosteriorDistr.mean(0) + kParam*sqrt(gaussianPosteriorDistr.sigma(0,0))
		val testTrainDerMtx:DenseMatrix[Double] = testTrainDerMatrix(derAfterFirstArg,testPoint,pointSet)
		val derAfterMean = testTrainDerMtx * alphaVec
		val derAfterVarFirst:DenseVector[Double] = derAfterFirstArg(testPoint,testPoint)
		val vAfterXDerMatrix:DenseMatrix[Double] = testTrainDerMtx * inversedL
		assert(vAfterXDerMatrix.rows == testPoint.length && vAfterXDerMatrix.cols == pointSet.rows)
		val derAfterVar:DenseVector[Double] = derAfterVarFirst - ((vAfterXDerMatrix * vMatrix.toDenseVector) :* 2.)
		val coeff:Double = (1/(2*sqrt(gaussianPosteriorDistr.sigma(0,0))))*kParam
		val ucbDer:DenseVector[Double] = derAfterMean + (derAfterVar :* coeff)
		(-ucbObjFunctionValue,ucbDer :* (-1.))
	  }
	}
	val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 30,m = 3)
	val optimalSolution = lbfgs.minimize(diffFunction,initPoint)
	(diffFunction(optimalSolution),optimalSolution)
  }

  private def testTrainDerMatrix(kernelDer:kernelDerivative,
								 testPoint:DenseVector[Double],trainingPoints:DenseMatrix[Double]):DenseMatrix[Double] = {
	val dim = testPoint.length
	val resultMatrix = DenseMatrix.zeros[Double](dim,trainingPoints.rows)
	for (col <- (0 until trainingPoints.rows)){
	  resultMatrix(::,col) := kernelDer(testPoint,trainingPoints(col,::).toDenseVector)
	}
	resultMatrix
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

  case class GPOInput(ranges:IndexedSeq[Range],mParam:Int,cParam:Int,kParam:Double)

}
