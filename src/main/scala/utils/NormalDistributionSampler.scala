package utils

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.distribution.MultivariateNormalDistribution

/**
 * Created by mjamroz on 10/03/14.
 */
class NormalDistributionSampler(means:DenseVector[Double],covs:DenseMatrix[Double]) {

  import NormalDistributionSampler._
  
  assert(means.length == covs.rows)
  assert(covs.cols == covs.rows)

  private val meanAsJavaArray = denseVecToArray(means)
  private val covAsJavaArray = denseMatrixTo2DimArray(covs)

  val multivariateSampler = new MultivariateNormalDistribution(meanAsJavaArray,covAsJavaArray)

  def sample:DenseVector[Double] = {
  	multivariateSampler.sample()
  }

}

object NormalDistributionSampler {
  
  implicit def denseVecToArray(vec:DenseVector[Double]):Array[Double] = {
	vec.data
  }
  
  implicit def denseMatrixTo2DimArray(m:DenseMatrix[Double]):Array[Array[Double]] = {
	(0 until m.rows).foldLeft(Array.ofDim[Double](m.rows,m.cols)){
	  case (arr,rowNum) => (0 until m.cols).foreach {colNum => arr(rowNum)(colNum) = m(rowNum,colNum)}
		arr
	}
  }

  implicit def arrayToDenseVector(arr:Array[Double]):DenseVector[Double] = {
	DenseVector(arr)
  }

}
