package dynamicalsystems.tests

import breeze.linalg.{DenseVector,DenseMatrix}
import dynamicalsystems.filtering.KalmanFilter
import dynamicalsystems.filtering.KalmanFilter.{StationaryTransitionMatrices, FilteringInput}

/**
 * Created by mjamroz on 30/03/14.
 */

/*
 * z_t - hidden state from R^4 - (z1_t,z2_t,z1_t_vel,z2_t_vel)
 * y_t - observed state from R^2 - (y1_t,y2_t)
 * z_t = A*z_(t-1) + e_t
 */
class ObjectTrackingTest {

  val timePeriod = 0.5

  val aTransMatrix:DenseMatrix[Double] = DenseMatrix((1.,0.,timePeriod,0.),(0.,1.,0.,timePeriod),(0.,0.,1.,0.),(0.,0.,0.,1.))
  val qNoise:DenseMatrix[Double] = DenseMatrix.eye[Double](4) :* 0.45
  val cTransMatrix:DenseMatrix[Double] = DenseMatrix((1.,0.,0.,0.),(0.,1.,0.,0.))
  val obs:DenseMatrix[Double] = DenseMatrix((8.,11.5,12.3,14.3,15.2,17.6,18.,18.7,20.2),
											(11.,10.1,8.6,10.5,9.6,11.,8.2,8.5,9.5))

  val rNoise:DenseMatrix[Double] = qNoise :* 1.4
  val initMean = DenseVector(9.2,11.2,3.,1.)
  val initCov = DenseMatrix.eye[Double](4) * 16.


  def test = {
	
	val kalmanFilter = new KalmanFilter
	val filteringInput = FilteringInput(observations = obs,initMean = initMean,initCov = initCov,
	  transMatrices = Right(StationaryTransitionMatrices(A = aTransMatrix,C = cTransMatrix,Q = qNoise,R = rNoise)))
	val filteringOutput = kalmanFilter.inferHiddenState(filteringInput,None,true)
	(0 until filteringOutput.hiddenMeans.cols).foreach{index =>
	  println(filteringOutput.hiddenMeans(::,index))
	}
	println(filteringOutput.logLikelihood)
  } 

}

object ObjectTrackingTest {

  def main(args:Array[String]) = {
	(new ObjectTrackingTest).test
  }
}

