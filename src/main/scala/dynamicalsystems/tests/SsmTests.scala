package dynamicalsystems.tests

import dynamicalsystems.filtering.SsmModel
import breeze.linalg.{DenseVector, DenseMatrix}
import utils.StatsUtils.GaussianDistribution

/**
 * Created by mjamroz on 11/05/14.
 */
object SsmTests {

  import dynamicalsystems.filtering.SsmExamples._

  class SinusoidalSsmTest extends SsmTest {

	override val rNoise: DenseMatrix[Double] = DenseMatrix((0.1*0.1))
	override val qNoise: DenseMatrix[Double] = rNoise
	override val ssmModel: SsmModel = new SinusoidalSsm
	override val resultFilePath: String = "ssm/sin"
  }

  class KitagawaSsmTest extends SsmTest {

	override val resultFilePath: String = "ssm/kit"
	override val qNoise: DenseMatrix[Double] = DenseMatrix((0.01*0.01))
	override val rNoise: DenseMatrix[Double] = DenseMatrix((0.2*0.2))
	override val ssmModel: SsmModel = new KitagawaSsm
	override val initHiddenStateDistr = GaussianDistribution(mean = DenseVector(0.),
					DenseMatrix((0.5*0.5)))
  }

  def main(args:Array[String]) = {

	val seqLength = 10
	val (sinusoidalSsmTest,kitSsmTest) = (new SinusoidalSsmTest,new KitagawaSsmTest)
	//sinusoidalSsmTest.run(seqLength)
	kitSsmTest.run(seqLength)
  }

}
