package dynamicalsystems.filtering

import breeze.linalg.{inv, DenseVector, DenseMatrix}

/**
 * Created by mjamroz on 29/03/14.
 */
class KalmanFilter {

  import KalmanFilter._
  import utils.StatsUtils._

  def inferHiddenState(filteringInput:FilteringInput,optionalInput:Option[OptionalInput],computeLL:Boolean):FilteringOutput = {

	val y = filteringInput.observations
	val (obsSpaceSize,tMax,hiddenSpaceSize) = (y.rows,y.cols,filteringInput.A(1).rows)
	val (hiddenMeans,hiddenCovs) = (DenseMatrix.zeros[Double](hiddenSpaceSize,tMax),new Array[transitionMatrix](tMax))

	val optInput = optionalInput match {
	  case Some(input) => input
	  case None =>
		val zeroTransitionMatrix = DenseMatrix.zeros[Double](hiddenSpaceSize,1)
		val arrWithZerosTransitionMatrix = new Array[transitionMatrix](tMax)
		(0 until tMax).foreach{index => arrWithZerosTransitionMatrix(index) = zeroTransitionMatrix}
		val (u,bMatrix,dMatrix) = (DenseMatrix.zeros[Double](tMax,1),arrWithZerosTransitionMatrix,arrWithZerosTransitionMatrix)
		OptionalInput(B = bMatrix,D = dMatrix,u = u)
	}

	for (t <- (0 until tMax).toStream){
	  val (prevM,prevCov) = (t == 0) match {
		case true => (filteringInput.initMean,filteringInput.initCov)
		case false => (hiddenMeans(::,t-1),hiddenCovs(t-1))
	  }
	  val (inferredMean,inferredCov,_) = kalmanUpdate(filteringInput,optInput,t,prevM,prevCov,computeLL)
	  hiddenMeans(::,t) := inferredMean
	  hiddenCovs(t) = inferredCov
	}
	FilteringOutput(hiddenMeans,hiddenCovs,None)
  }

  private def kalmanUpdate(filteringInput:FilteringInput,optionalInput:OptionalInput,t:Int,
						   prevMean:DenseVector[Double],prevCov:DenseMatrix[Double],computeLL:Boolean):
  	(DenseVector[Double],DenseMatrix[Double],Option[Double]) = {

	val (y,aMatrix,cMatrix,qNoise,rNoise) =
	  (filteringInput.observations(::,t),filteringInput.A(t),filteringInput.C(t),filteringInput.Q(t),filteringInput.R(t))
	val (u,bMatrix,dMatrix) = (optionalInput.u(::,t),optionalInput.B(t),optionalInput.D(t))
	val (mPred:DenseVector[Double],covPred:DenseMatrix[Double]) = (t == 0) match {
	  case true => (prevMean + (bMatrix * u),prevCov)
	  case false =>
		(aMatrix*prevMean + bMatrix*u,((aMatrix * prevCov) * aMatrix.t) + qNoise)
	}
	val stateSize = prevMean.length
	val error:DenseVector[Double] = y - (cMatrix * mPred) - (dMatrix * u)
	val sMatrix:DenseMatrix[Double] = ((cMatrix * covPred) * cMatrix.t) + rNoise
	val sMatrixInv:DenseMatrix[Double] = inv(sMatrix)
	val kalmanGainMtx:DenseMatrix[Double] = (covPred * cMatrix.t) * sMatrixInv
	val temp1:DenseMatrix[Double] = (DenseMatrix.eye[Double](stateSize) - (kalmanGainMtx * cMatrix))
	val logLikelihood = computeLL match {
	  case true =>
		Some(logGaussianDensity(error,means = DenseVector.zeros[Double](error.length),covs = sMatrix))
	  case false =>	None
	}
	(mPred + (kalmanGainMtx * error),temp1 * covPred,logLikelihood)
  }

}


object KalmanFilter{

  type transitionMatrix = DenseMatrix[Double]

  /*
   * Z_t = A*Z_(t-1) + B*u_t + Q_t
   * Y_t = C*Z_t + D*u_t + R_t
   * p(z1) = N(initMean,initCov)
   * observations(::,t) - observed state in step t
   */
  case class FilteringInput(observations:DenseMatrix[Double],A:Array[transitionMatrix],
							C:Array[transitionMatrix],Q:Array[transitionMatrix],R:Array[transitionMatrix],
							 initMean:DenseVector[Double],initCov:DenseMatrix[Double])


  case class OptionalInput(B:Array[transitionMatrix],D:Array[transitionMatrix],u:DenseMatrix[Double])

  /*
   * p(z_t | y_1:t) = N(hiddenMean(::,t),hiddenCovs(t))
   */
  case class FilteringOutput(hiddenMeans:DenseMatrix[Double],hiddenCovs:Array[transitionMatrix],
							 logLikelihood:Option[Double])

}
