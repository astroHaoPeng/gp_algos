package dynamicalsystems.filtering

import breeze.linalg.{inv, DenseVector, DenseMatrix}

/**
 * Created by mjamroz on 29/03/14.
 */
class KalmanFilter {

  import KalmanFilter._
  import utils.StatsUtils._
  import SsmTypeDefinitions._

  def inferHiddenState(filteringInput:FilteringInput,
					   optionalInput:Option[OptionalInput],computeLL:Boolean):FilteringOutput = {

	filteringInput.transMatrices match {
	  case Left(transitionMatrices) => inferHiddenStateNonStat(filteringInput,optionalInput,computeLL)
	  case Right(stationaryTransMatrices) =>
		val nonStatTransMatrices = stationaryModelToSeq(filteringInput.observations.cols,
		stationaryTransMatrices)
		inferHiddenStateNonStat(filteringInput.copy(transMatrices = Left(nonStatTransMatrices)),optionalInput,computeLL)
	}
  }

  //TODO validate matrices dimensions
  private def inferHiddenStateNonStat(filteringInput:FilteringInput,optionalInput:Option[OptionalInput],computeLL:Boolean):FilteringOutput = {

	val y = filteringInput.observations
	val (obsSpaceSize,tMax,hiddenSpaceSize) = (y.rows,y.cols,filteringInput.transMatrices.left.get.A(1).rows)
	val (hiddenMeans,hiddenCovs) = (DenseMatrix.zeros[Double](hiddenSpaceSize,tMax),new Array[transitionMatrix](tMax))

	val optInput = optionalInput match {
	  case Some(input) => input
	  case None =>
		generateOptionalInput(tMax,hiddenSpaceSize,obsSpaceSize)
	}
	var logLikelihood:Option[Double] = if (computeLL){Some(0.)} else {None}
	for (t <- (0 until tMax).toStream){
	  val (prevM,prevCov) = (t == 0) match {
		case true => (filteringInput.initMean,filteringInput.initCov)
		case false => (hiddenMeans(::,t-1),hiddenCovs(t-1))
	  }
	  val (inferredMean,inferredCov,ll) = kalmanUpdate(filteringInput,optInput,t,prevM,prevCov,computeLL)
	  hiddenMeans(::,t) := inferredMean
	  hiddenCovs(t) = inferredCov
	  logLikelihood = ll.map(_ + logLikelihood.get)
	}
	FilteringOutput(hiddenMeans,hiddenCovs,logLikelihood)
  }

  private def kalmanUpdate(filteringInput:FilteringInput,optionalInput:OptionalInput,t:Int,
						   prevMean:DenseVector[Double],prevCov:DenseMatrix[Double],computeLL:Boolean):
  	(DenseVector[Double],DenseMatrix[Double],Option[Double]) = {

	val transMatrices = filteringInput.transMatrices.left.get
	val (y,aMatrix,cMatrix,qNoise,rNoise) =
	  (filteringInput.observations(::,t),transMatrices.A(t),transMatrices.C(t),transMatrices.Q(t),transMatrices.R(t))
	val (u,bMatrix,dMatrix) = (optionalInput.u(::,t),optionalInput.B(t),optionalInput.D(t))
	val (obsSpaceSize,hiddenSpaceSize) = (y.length,aMatrix.rows)
	assert(prevMean.length == hiddenSpaceSize && prevCov.rows == hiddenSpaceSize && prevCov.cols == hiddenSpaceSize)
	assert(y.length == obsSpaceSize)
	assert(bMatrix.rows == hiddenSpaceSize && bMatrix.cols == u.length)
	assert(dMatrix.rows == obsSpaceSize && dMatrix.cols == u.length)
	val (mPred:DenseVector[Double],covPred:DenseMatrix[Double]) = (t == 0) match {
	  case true => (prevMean + (bMatrix * u),prevCov)
	  case false =>
		(aMatrix*prevMean + bMatrix*u,((aMatrix * prevCov) * aMatrix.t) + qNoise)
	}
	val error:DenseVector[Double] = y - (cMatrix * mPred) - (dMatrix * u)
	val sMatrix:DenseMatrix[Double] = ((cMatrix * covPred) * cMatrix.t) + rNoise
	val sMatrixInv:DenseMatrix[Double] = inv(sMatrix)
	val kalmanGainMtx:DenseMatrix[Double] = (covPred * cMatrix.t) * sMatrixInv
	val temp1:DenseMatrix[Double] = (DenseMatrix.eye[Double](hiddenSpaceSize) - (kalmanGainMtx * cMatrix))
	val logLikelihood = computeLL match {
	  case true =>
		Some(logGaussianDensity(error,means = DenseVector.zeros[Double](error.length),covs = sMatrix))
	  case false =>	None
	}
	(mPred + (kalmanGainMtx * error),temp1 * covPred,logLikelihood)
  }

  private def generateOptionalInput(tMax:Int,hiddenSpaceSize:Int,obsSpaceSize:Int):OptionalInput = {

	val (zeroBTransMatrix,zeroDTransMatrix) = (DenseMatrix.zeros[Double](hiddenSpaceSize,1),
	  DenseMatrix.zeros[Double](obsSpaceSize,1))
	val (arrWithZerosTransitionMatrix,arrWithZerosTransitionMatrix1) = (new Array[transitionMatrix](tMax),
	  new Array[transitionMatrix](tMax))
	(0 until tMax).foreach{index => arrWithZerosTransitionMatrix(index) = zeroBTransMatrix
	  arrWithZerosTransitionMatrix1(index) = zeroDTransMatrix}
	val (u,bMatrix,dMatrix) = (DenseMatrix.zeros[Double](1,tMax),
	  arrWithZerosTransitionMatrix,arrWithZerosTransitionMatrix1)
	OptionalInput(B = bMatrix,D = dMatrix,u = u)
  }

  private def stationaryModelToSeq(tMax:Int,stationaryMatrices:StationaryTransitionMatrices):TransitionsMatrices = {
	val A:Array[transitionMatrix] = new Array[transitionMatrix](tMax)
	val C:Array[transitionMatrix] = new Array[transitionMatrix](tMax)
	val Q:Array[transitionMatrix] = new Array[transitionMatrix](tMax)
	val R:Array[transitionMatrix] = new Array[transitionMatrix](tMax)

	(0 until A.length).foreach(C(_) = stationaryMatrices.C)
	(0 until C.length).foreach(A(_) = stationaryMatrices.A)
	(0 until Q.length).foreach(Q(_) = stationaryMatrices.Q)
	(0 until R.length).foreach(R(_) = stationaryMatrices.R)
	TransitionsMatrices(A = A,C = C,Q = Q,R = R)
  }

}


object KalmanFilter{

  import SsmTypeDefinitions._

  /*
   * Z_t = A*Z_(t-1) + B*u_t + Q_t
   * Y_t = C*Z_t + D*u_t + R_t
   * p(z1) = N(initMean,initCov)
   * observations(::,t) - observed state in step t
   */

  case class FilteringInput(transMatrices:Either[TransitionsMatrices,StationaryTransitionMatrices],
							observations:DenseMatrix[Double],
								initMean:DenseVector[Double],initCov:DenseMatrix[Double])

  case class TransitionsMatrices(A:Array[transitionMatrix],C:Array[transitionMatrix],
								 Q:Array[transitionMatrix],R:Array[transitionMatrix])

  case class StationaryTransitionMatrices(A:transitionMatrix,C:transitionMatrix,
										  Q:transitionMatrix,R:transitionMatrix)

  case class OptionalInput(B:Array[transitionMatrix],D:Array[transitionMatrix],u:DenseMatrix[Double])

  /*
   * p(z_t | y_1:t) = N(hiddenMean(::,t),hiddenCovs(t))
   */
  case class FilteringOutput(hiddenMeans:DenseMatrix[Double],hiddenCovs:Array[transitionMatrix],
							 logLikelihood:Option[Double])

}
