package dynamicalsystems.tests

import scala.collection.immutable.NumericRange
import dynamicalsystems.filtering.UnscentedKalmanFilter.UnscentedTransformParams
import org.slf4j.LoggerFactory

/**
 * Created by mjamroz on 13/05/14.
 */
class MeshUkfParamsEvaluator {

  import gp.classification.MeshHyperParamsLogLikelihoodEvaluator._

  val logger = LoggerFactory.getLogger(this.getClass)

  def evaluate(ukfParamsRanges: IndexedSeq[NumericRange[Double]])(ukfEvalFunction: UnscentedTransformParams => Double)
  : HyperParamsMeshValues = {

	require(ukfParamsRanges.length == 3, "Exactly three ranges are required")
	val result = new HyperParamsMeshValues
	evaluateMesh(ukfParamsRanges,result)(ukfEvalFunction)
	result
  }

  def evaluateMesh(ranges: IndexedSeq[NumericRange[Double]], result: HyperParamsMeshValues)
				  (ukfEvalFunction: UnscentedTransformParams => Double) = {

	for (alpha <- ranges(0); beta <- ranges(1); kappa <- ranges(2)) {
	  val ukfParams = UnscentedTransformParams(alpha = alpha, beta = beta, kappa = kappa)
	  val functionVal = ukfEvalFunction(ukfParams)
	  logger.info(s"Evaluate log likelihood for = $ukfParams, value = $functionVal")
	  result.addResult(ukfParams.toVector,functionVal)
	}

  }

}
