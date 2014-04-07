package dynamicalsystems.filtering

import breeze.linalg.{inv, DenseMatrix, DenseVector}
import breeze.numerics.sqrt

/**
 * Created by mjamroz on 01/04/14.
 */
class UnscentedKalmanFilter {

  import utils.StatsUtils._
  import UnscentedKalmanFilter._
  import KalmanFilter._

  def inferHiddenState(input: UnscentedFilteringInput, params: Option[UnscentedTransformParams],
							   computeLL: Boolean): FilteringOutput = {

	val unscentedParams = params.getOrElse(UnscentedTransformParams())
	val y: DenseMatrix[Double] = input.observations
	val (obsSpaceSize, tMax, hiddenSpaceSize) = (y.rows, y.cols, input.initMean.length)
	val optionalInput = input.u.getOrElse(DenseMatrix.zeros[Double](1, tMax))
	val (hiddenMeans, hiddenCovs) = (DenseMatrix.zeros[Double](hiddenSpaceSize, tMax), new Array[transitionMatrix](tMax))

	for (t <- (0 until tMax).toStream) {
	  val (probab_z_t_given_prev_y: GaussianDistribution, sigmaWithoutQNoise: DenseMatrix[Double]) = (t == 0) match {
		case true => GaussianDistribution(mean = input.initMean, sigma = input.initCov)
		case false =>
		  val u_t = optionalInput(::, t)
		  val mappingFunc: DenseVector[Double] => DenseVector[Double] = {
			point => input.transitionModelFunc(u_t, point, t)
		  }
		  val prevGaussianDistr = GaussianDistribution(mean = hiddenMeans(::, t - 1), sigma = hiddenCovs(t - 1))
		  val distrAfterTransform = unscentedTransform(prevGaussianDistr, unscentedParams)(mappingFunc)
		  (GaussianDistribution(mean = distrAfterTransform.mean, sigma = distrAfterTransform.sigma + input.qNoise(t)),
			prevGaussianDistr.sigma)
	  }
	  val obsMappingFunc: DenseVector[Double] => DenseVector[Double] = {
		point => input.observationModelFunc(point, t)
	  }
	  val distrAfterTransform = unscentedTransform(probab_z_t_given_prev_y, unscentedParams)(obsMappingFunc)
	  val probab_y_t_given_z_t = GaussianDistribution(mean = distrAfterTransform.mean,
		sigma = distrAfterTransform.sigma + input.rNoise(t))
	  val sMatrix = probab_y_t_given_z_t.sigma
	  val sMatrixInv: DenseMatrix[Double] = inv(sMatrix)
	  val kalmanGainMatrix: DenseMatrix[Double] = sigmaWithoutQNoise * sMatrixInv
	  val hiddenMean:DenseVector[Double] = probab_z_t_given_prev_y.mean +
	  	(kalmanGainMatrix * (input.observations(::,t) - probab_y_t_given_z_t.mean))
	  val hiddenCov:DenseMatrix[Double] = probab_z_t_given_prev_y.sigma -
	  	((kalmanGainMatrix * sMatrix) * kalmanGainMatrix.t)
	  hiddenMeans(::,t) := hiddenMean
	  hiddenCovs(t) = hiddenCov
	}
	FilteringOutput(hiddenMeans = hiddenMeans,hiddenCovs = hiddenCovs,logLikelihood = None)
  }

  def unscentedTransform(normalDistribution: GaussianDistribution,
								 params: UnscentedTransformParams)(func: DenseVector[Double] => DenseVector[Double]): GaussianDistribution = {

	val (mean, covMatrix) = (normalDistribution.mean, normalDistribution.sigma)
	val dim = normalDistribution.dim
	val (sigmaPoints, lambda) = (DenseMatrix.zeros[Double](2 * dim + 1, dim), params.alpha * params.alpha * (params.d + params.kappa))
	sigmaPoints(0, ::) := mean
	for (row <- (0 until dim)) {
	  val sqrtCoeff: DenseVector[Double] = sqrt(covMatrix(::, row) :* (params.d + lambda))
	  sigmaPoints(row + 1, ::) := mean + sqrtCoeff
	  sigmaPoints(row + 1 + dim, ::) := mean - sqrtCoeff
	}

	val (w_0_m, w_0_c, w_i_c) = (lambda / (params.d + lambda),
	  (lambda / (params.d + lambda)) + (1 - params.alpha * params.alpha + params.beta), 1 / (2 * (params.d + lambda)))
	val firstSigmaPointTransformed:DenseVector[Double] = func(sigmaPoints(0, ::).toDenseVector)
	val firstMeanCoeff: DenseVector[Double] = firstSigmaPointTransformed :* w_0_m
	val finalMean: DenseVector[Double] = (1 until (2 * dim + 1)).foldLeft(firstMeanCoeff) {
	  case (tempMean, index) =>
		val transformedSigmaPoint = func(sigmaPoints(index, ::).toDenseVector)
		tempMean :+ (transformedSigmaPoint :* w_i_c)
	}
	val diff:DenseVector[Double] = sigmaPoints(0, ::).toDenseVector - finalMean
	var finalCovMatrix: DenseMatrix[Double] = (diff * diff.t) :* w_0_c
	finalCovMatrix = (1 until (2 * dim + 1)).foldLeft(finalCovMatrix) {
	  case (tempCov, index) =>
		val diff: DenseVector[Double] = sigmaPoints(index, ::).toDenseVector - finalMean
		tempCov :+ ((diff * diff.t) :* w_i_c)
	}
	GaussianDistribution(mean = finalMean.toDenseVector, sigma = finalCovMatrix)
  }

}


object UnscentedKalmanFilter {

  import KalmanFilter._

  /*1 - optional input u_t, 2 - previous state z_(t-1), 3 - time step, result - new state z_t */
  type transitionFunc = (DenseVector[Double], DenseVector[Double], Int) => DenseVector[Double]

  /*1 - hidden state z_t, 2 - time step, result - observation state y_t*/
  type observationFunc = (DenseVector[Double], Int) => DenseVector[Double]

  case class UnscentedTransformParams(d: Double, alpha: Double, beta: Double, kappa: Double)

  object UnscentedTransformParams {

	def apply():UnscentedTransformParams = defaultParams

	def defaultParams:UnscentedTransformParams = UnscentedTransformParams(d = 1., alpha = 1., beta = 0., kappa = 2.)

  }

  case class UnscentedFilteringInput(transitionModelFunc: transitionFunc,
									 observationModelFunc: observationFunc,
									 observations: DenseMatrix[Double],
									 u: Option[DenseMatrix[Double]],
									 initMean: DenseVector[Double], initCov: DenseMatrix[Double],
									 qNoise: Array[transitionMatrix], rNoise: Array[transitionMatrix])

}