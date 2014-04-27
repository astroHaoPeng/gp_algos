package dynamicalsystems.filtering

import breeze.linalg.{DenseMatrix, DenseVector}
import utils.StatsUtils.GaussianDistribution

/**
 * Created by mjamroz on 24/04/14.
 */
object SsmTypeDefinitions {

  type transitionMatrix = DenseMatrix[Double]

  /*1 - optional input u_t, 2 - previous state z_(t-1), 3 - time step, result - new state z_t */
  type transitionFunc = (DenseVector[Double], DenseVector[Double], Int) => DenseVector[Double]

  /*1 - hidden state z_t, 2 - time step, result - observation state y_t*/
  type observationFunc = (DenseVector[Double], Int) => DenseVector[Double]

  case class SeriesGenerationData(qNoise:Array[transitionMatrix],
								  rNoise:Array[transitionMatrix],
								  initHiddenState:Either[DenseVector[Double],GaussianDistribution])

}
