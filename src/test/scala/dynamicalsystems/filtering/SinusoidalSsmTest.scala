package dynamicalsystems.filtering

import org.scalatest.WordSpec
import dynamicalsystems.filtering.SsmExamples.SinusoidalSsm
import dynamicalsystems.filtering.SsmTypeDefinitions.SeriesGenerationData
import breeze.linalg.DenseMatrix
import utils.StatsUtils.GaussianDistribution
import dynamicalsystems.tests.SsmTestingUtils

/**
 * Created by mjamroz on 25/04/14.
 */
class SinusoidalSsmTest extends WordSpec with SsmTestingUtils{

  val ssmSampler = new SinusoidalSsm
  val rNoise:DenseMatrix[Double] = DenseMatrix((0.1*0.1))
  val qNoise:DenseMatrix[Double] = rNoise

  "Sinusoidal state space model" should {

	"generate a sequence of observations" in {

	  	val seqLength = 10;
	  	val genData = SeriesGenerationData(
		  initHiddenState = Right(GaussianDistribution.standard))
		val (hidden,obs) = ssmSampler.generateSeries(seqLength,genData)
	    assert(hidden.rows == 1 && obs.rows == 1)
	  	assert(hidden.cols == seqLength && obs.cols == seqLength)
	}

  }

}
