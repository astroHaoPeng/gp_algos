package dynamicalsystems.filtering

import breeze.linalg.{cholesky, inv, DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import utils.StatsUtils.GaussianDistribution
import gp.optimization.GPOptimizer
import gp.optimization.GPOptimizer.GPOInput
import utils.StatsUtils

/**
 * Created by mjamroz on 01/04/14.
 */
class UnscentedKalmanFilter(gpOptimizer:GPOptimizer) {

  import utils.StatsUtils._
  import UnscentedKalmanFilter._
  import KalmanFilter._
  import SsmTypeDefinitions._

  val ukfParamRange = 0 to 10
  /*Sth bigger than Double.minVal, but still reasonable value that can be used in computations*/
  val veryLowValue = -scala.math.pow(10,6)

  def inferHiddenState(input: UnscentedFilteringInput, params: Option[UnscentedTransformParams],
					   computeLL: Boolean = true): FilteringOutput = {

	val unscentedParams = params.getOrElse(UnscentedTransformParams())
	val y: DenseMatrix[Double] = input.observations
	val (obsSpaceSize, tMax, hiddenSpaceSize) = (y.rows, y.cols, input.initMean.length)
	var ll:Option[Double] = if (computeLL){Some(0.)} else {None}
	val optionalInput = input.u.getOrElse(DenseMatrix.zeros[Double](1, tMax))
	val (hiddenMeans, hiddenCovs) = (DenseMatrix.zeros[Double](hiddenSpaceSize, tMax), new Array[transitionMatrix](tMax))
	hiddenMeans(::, 0) := input.initMean
	hiddenCovs(0) = input.initCov
	val inferenceContext = UkfInferenceContext(iteration = 0,hiddenMeans = hiddenMeans,
	  hiddenCovs = hiddenCovs,firstTransformFromIteration = null,secondTransformFromIteration = null)

	for (t <- (1 until tMax).toStream) {

	  val u_t = optionalInput(::, t)
	  val hiddenMappingFunc: DenseVector[Double] => DenseVector[Double] = {
		point => input.ssmModel.transitionFuncImpl(u_t, point, t)
	  }
	  val prevGaussianDistr = GaussianDistribution(mean = hiddenMeans(::, t - 1), sigma = hiddenCovs(t - 1))
	  val firstUTransformOut: UnscentedTransformOutput = unscentedTransform(prevGaussianDistr, unscentedParams)(hiddenMappingFunc)
	  val qNoise = input.qNoise(inferenceContext.copy(iteration = t,firstTransformFromIteration = firstUTransformOut))
	  val probab_z_t_given_prev_y: GaussianDistribution = firstUTransformOut.
		distribution.copy(sigma = firstUTransformOut.distribution.sigma + qNoise)
	  val obsMappingFunc: DenseVector[Double] => DenseVector[Double] = {
		point => input.ssmModel.observationFuncImpl(point, t)
	  }
	  val secondUTransformOut = unscentedTransform(probab_z_t_given_prev_y, unscentedParams)(obsMappingFunc)
	  val distrAfterTransform = secondUTransformOut.distribution
	  val rNoise = input.rNoise(inferenceContext.copy(iteration = t,
		firstTransformFromIteration = firstUTransformOut,secondTransformFromIteration = secondUTransformOut))
	  val probab_y_t_given_z_t: GaussianDistribution = GaussianDistribution(mean = distrAfterTransform.mean,
		sigma = distrAfterTransform.sigma + rNoise)

	  val (zTransformed,yTransformed,weights) =
		(firstUTransformOut.transformedSigmaPoints,secondUTransformOut.transformedSigmaPoints,firstUTransformOut.weights)
	  val initZYCovMatrix:DenseMatrix[Double] =
		((zTransformed(0,::).t - probab_z_t_given_prev_y.mean) * (yTransformed(0,::).t - probab_y_t_given_z_t.mean).t) :* weights._2
	  val z_y_covMatrix:DenseMatrix[Double] = (1 until (2*hiddenSpaceSize+1)).
		foldLeft[DenseMatrix[Double]](initZYCovMatrix){
		case (currentCov,index) =>
			val update:DenseMatrix[Double] =
			  ((zTransformed(index,::).t - probab_z_t_given_prev_y.mean) * (yTransformed(index,::).t - probab_y_t_given_z_t.mean).t) :* weights._3
			currentCov :+ update
	  }
	  assert(z_y_covMatrix.rows == hiddenSpaceSize && z_y_covMatrix.cols == obsSpaceSize)

	  val sMatrix = probab_y_t_given_z_t.sigma
	  val sMatrixInv: DenseMatrix[Double] = inv(sMatrix)
	  val kalmanGainMatrix: DenseMatrix[Double] = z_y_covMatrix * sMatrixInv
	  val hiddenMean: DenseVector[Double] = probab_z_t_given_prev_y.mean +
		(kalmanGainMatrix * (y(::, t) - probab_y_t_given_z_t.mean))
	  val hiddenCov: DenseMatrix[Double] = probab_z_t_given_prev_y.sigma -
		((kalmanGainMatrix * sMatrix) * kalmanGainMatrix.t)
	  hiddenMeans(::, t) := hiddenMean
	  hiddenCovs(t) = hiddenCov
	  ll = ll.map(_ + marginalLogLikelihood(y(::, t),probab_y_t_given_z_t))
	}
	FilteringOutput(hiddenMeans = hiddenMeans, hiddenCovs = hiddenCovs, logLikelihood = ll)
  }

  def unscentedTransform(normalDistribution: GaussianDistribution,
						 params: UnscentedTransformParams)(func: DenseVector[Double] => DenseVector[Double]): UnscentedTransformOutput = {

	val (mean, covMatrix, d) = (normalDistribution.mean, normalDistribution.sigma,normalDistribution.dim)
	val lowerTriangular:DenseMatrix[Double] = cholesky(covMatrix)
	val dim = normalDistribution.dim
	val (sigmaPoints, lambda) = (DenseMatrix.zeros[Double](2 * dim + 1, dim), params.alpha * params.alpha * (d + params.kappa) - d)
	sigmaPoints(0, ::) := mean.t
	for (col <- (0 until dim)) {
	  val sqrtCoeff: DenseVector[Double] = lowerTriangular(::, col) :* math.sqrt(d + lambda)
	  sigmaPoints(col + 1, ::) := (mean + sqrtCoeff).t
	  sigmaPoints(col + 1 + dim, ::) := (mean - sqrtCoeff).t
	}

	val (w_0_m, w_0_c, w_i_c) = (lambda / (d + lambda),
	  (lambda / (d + lambda)) + (1 - params.alpha * params.alpha + params.beta), 1 / (2 * (d + lambda)))
	val firstSigmaPointTransformed: DenseVector[Double] = func(sigmaPoints(0, ::).t)
	val transformedSigmaPoints: DenseMatrix[Double] = DenseMatrix.zeros[Double](2 * dim + 1, firstSigmaPointTransformed.length)
	transformedSigmaPoints(0, ::) := firstSigmaPointTransformed.t
	val firstMeanCoeff: DenseVector[Double] = firstSigmaPointTransformed :* w_0_m
	val finalMean: DenseVector[Double] = (1 until (2 * dim + 1)).foldLeft(firstMeanCoeff) {
	  case (tempMean, index) =>
		val transformedSigmaPoint = func(sigmaPoints(index, ::).t)
		transformedSigmaPoints(index, ::) := transformedSigmaPoint.t
		tempMean :+ (transformedSigmaPoint :* w_i_c)
	}
	val diff: DenseVector[Double] = transformedSigmaPoints(0, ::).t - finalMean
	var finalCovMatrix: DenseMatrix[Double] = (diff * diff.t) :* w_0_c
	finalCovMatrix = (1 until (2 * dim + 1)).foldLeft(finalCovMatrix) {
	  case (tempCov, index) =>
		val diff: DenseVector[Double] = transformedSigmaPoints(index, ::).t - finalMean
		tempCov :+ ((diff * diff.t) :* w_i_c)
	}
	val distr = GaussianDistribution(mean = finalMean, sigma = finalCovMatrix)
	UnscentedTransformOutput(distribution = distr, weights = (w_0_m, w_0_c, w_i_c),
	  sigmaPoints = sigmaPoints, transformedSigmaPoints = transformedSigmaPoints)
  }

  def inferWithUkfOptimWrtToMarginall(input:UnscentedFilteringInput,
								 initParams:Option[UnscentedTransformParams],rangeForParam:Range=ukfParamRange):FilteringOutput = {

	val objFunction:optimization.Optimization.objectiveFunction = {
	  point:Array[Double] =>
		val unscentedParams = UnscentedTransformParams.fromVector(point)
		val out = inferHiddenState(input,Some(unscentedParams),true)
	  	/*We want to minimize negative log likelihood */
	  	if (out.logLikelihood.get == Double.NegativeInfinity){Double.MinValue}
	  	else if (out.logLikelihood.get == Double.PositiveInfinity){Double.MaxValue}
	  	else {out.logLikelihood.get}
	}
	val gpoInput = GPOInput(mParam = 50,cParam = 10,kParam = 2.,
	  ranges = IndexedSeq(rangeForParam,rangeForParam,rangeForParam))
	val (optimizedParams,_) = gpOptimizer.maximize(objFunction,gpoInput)
	inferHiddenState(input,Some(UnscentedTransformParams.fromVector(optimizedParams)),true)
  }
  
  def inferWithUkfOptimWrtToNll(input:UnscentedFilteringInput,
								initParams:Option[UnscentedTransformParams],hidden:DenseMatrix[Double],rangeForParam:Range=ukfParamRange) = {
	val objFunction:optimization.Optimization.objectiveFunction = {
	  point:Array[Double] =>
		val unscentedParams = UnscentedTransformParams.fromVector(point)
		val out = inferHiddenState(input,Some(unscentedParams),true)
	  	val nll = nllOfHiddenData(trueHiddenStates = hidden,
		  hiddenMeans = out.hiddenMeans,hiddenCovs = out.hiddenCovs)
		/*We want to minimize negative log likelihood */
		if (nll == Double.NegativeInfinity){veryLowValue}
		else if (nll == Double.PositiveInfinity){-veryLowValue}
		else {nll}
	}
	val gpoInput = getGpoInput(rangeForParam)
	val (optimizedParams,_) = gpOptimizer.minimize(objFunction,gpoInput)
	inferHiddenState(input,Some(UnscentedTransformParams.fromVector(optimizedParams)),true)
  }

  protected def getGpoInput(rangeForParam:Range):GPOInput = {
	GPOInput(mParam = 50,cParam = 10,kParam = 2.,
	  ranges = IndexedSeq(rangeForParam,rangeForParam,rangeForParam))
  }

  private def marginalLogLikelihood(y_t:DenseVector[Double],probab_y_t_given_z_t:GaussianDistribution):Double = {
	val returnVal = logGaussianDensity(at = y_t,means = probab_y_t_given_z_t.mean,covs = probab_y_t_given_z_t.sigma)
  	returnVal
  }

}


object UnscentedKalmanFilter {

  import SsmTypeDefinitions._

  case class UnscentedTransformParams(alpha: Double, beta: Double, kappa: Double){

	def toVector:Array[Double] = Array[Double](alpha,beta,kappa)

  }

  object UnscentedTransformParams {

	def apply(): UnscentedTransformParams = defaultParams

	def defaultParams: UnscentedTransformParams = UnscentedTransformParams(alpha = 1., beta = 0., kappa = 2.)

	def fromVector(arr:Array[Double]):UnscentedTransformParams =
	  UnscentedTransformParams(alpha = arr(0),beta = arr(1),kappa = arr(2))
  }

  /*weights - (w_0_m,w_0_c,w_i_c)*/
  case class UnscentedTransformOutput(distribution: GaussianDistribution,
									  weights: (Double, Double, Double),
									  sigmaPoints: DenseMatrix[Double], transformedSigmaPoints: DenseMatrix[Double])

  case class UnscentedFilteringInput(ssmModel:SsmModel,
									 observations: DenseMatrix[Double],
									 u: Option[DenseMatrix[Double]],
									 initMean: DenseVector[Double], initCov: DenseMatrix[Double],
									 qNoise: noiseComputationFunc, rNoise: noiseComputationFunc )

  object UnscentedFilteringInput {

	def classicUkfNoise(qNoise:Array[transitionMatrix],rNoise:Array[transitionMatrix]):
		(noiseComputationFunc,noiseComputationFunc) = {
	  val qNoiseFunc:noiseComputationFunc = {context => qNoise(context.iteration)}
	  val rNoiseFunc:noiseComputationFunc = {context => rNoise(context.iteration)}
	  (qNoiseFunc,rNoiseFunc)
	}

  }


  case class UkfInferenceContext(iteration:Int,hiddenMeans:DenseMatrix[Double],
								 hiddenCovs:Array[SsmTypeDefinitions.transitionMatrix],
								 firstTransformFromIteration:UnscentedTransformOutput,
								 secondTransformFromIteration:UnscentedTransformOutput)

  type noiseComputationFunc = UkfInferenceContext => DenseMatrix[Double]


}