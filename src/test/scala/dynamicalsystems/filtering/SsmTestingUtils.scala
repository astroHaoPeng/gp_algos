package dynamicalsystems.filtering

import breeze.linalg.DenseMatrix

/**
 * Created by mjamroz on 25/04/14.
 */
trait SsmTestingUtils {

  def cloneMatrix(matrix:DenseMatrix[Double],howMany:Int):Array[DenseMatrix[Double]] = {
	val resultArr = new Array[DenseMatrix[Double]](howMany)
	(0 until howMany).foreach(indx => resultArr(indx) = matrix)
	resultArr
  }

}
