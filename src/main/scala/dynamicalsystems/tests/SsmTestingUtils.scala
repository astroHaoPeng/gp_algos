package dynamicalsystems.tests

import breeze.linalg.DenseMatrix
import utils.StatsUtils.GaussianDistribution

/**
 * Created by mjamroz on 25/04/14.
 */
trait SsmTestingUtils {

  val resourcePathPrefix = "src/main/resources"
  def cloneMatrix(matrix:DenseMatrix[Double],howMany:Int):Array[DenseMatrix[Double]] = {
	val resultArr = new Array[DenseMatrix[Double]](howMany)
	(0 until howMany).foreach(indx => resultArr(indx) = matrix)
	resultArr
  }

  case class SsmTestingResult(hiddenStates:Seq[GaussianDistribution],ll:Double,mse:Double)

}
