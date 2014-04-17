package utils

import breeze.stats.distributions.Gaussian
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import breeze.numerics.log

/**
 * Created by mjamroz on 13/03/14.
 */
object StatsUtils {

  val gaussianHelper = Gaussian(mu = 0,sigma = 1)

  def dnorm(x:Double) = gaussianHelper.pdf(x)

  def pnorm(x:Double) = gaussianHelper.cdf(x)

  case class GaussianDistribution(mean:DenseVector[Double],sigma:DenseMatrix[Double]){
	def dim:Int = mean.length
  }

  class NormalDistributionSampler(normalDistr:GaussianDistribution) {

	import NormalDistributionSampler._

	val (mean,covs) = (normalDistr.mean,normalDistr.sigma)
	
	require(mean.length == covs.rows)
	require(covs.cols == covs.rows)

	private val meanAsJavaArray = denseVecToArray(mean)
	private val covAsJavaArray = denseMatrixTo2DimArray(covs)

	val multivariateSampler = new MultivariateNormalDistribution(meanAsJavaArray,covAsJavaArray)

	def sample:DenseVector[Double] = {
	  multivariateSampler.sample()
	}

  }

  def gaussianDensity(at:DenseVector[Double],means:DenseVector[Double],covs:DenseMatrix[Double]):Double = {

	import NormalDistributionSampler._

	val multiVariateNormalDistr = new MultivariateNormalDistribution(means,covs)
	multiVariateNormalDistr.density(at.data)
  }

  def logGaussianDensity(at:DenseVector[Double],means:DenseVector[Double],covs:DenseMatrix[Double]):Double = {
	log(gaussianDensity(at,means,covs))
  }

  /*data(i,::) - ith sample*/
  def meanAndVarOfData(data:DenseMatrix[Double]):(DenseVector[Double],DenseMatrix[Double]) = {
	val sumOfSamples:DenseVector[Double] = (0 until data.rows).foldLeft(DenseVector.zeros[Double](data.cols)){
	  case (mean,index) => mean + data(index,::).toDenseVector
	}
	val mean:DenseVector[Double] = sumOfSamples :/ data.rows.toDouble
	val covMatrix:DenseMatrix[Double] = (0 until data.rows).foldLeft(DenseMatrix.zeros[Double](data.cols,data.cols)){
	  case (cov,index) =>
		val diff:DenseVector[Double] = (data(index,::).toDenseVector - mean)
		cov :+ (diff * diff.t)
	}
	(mean,covMatrix :/ data.rows.toDouble)
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

}
