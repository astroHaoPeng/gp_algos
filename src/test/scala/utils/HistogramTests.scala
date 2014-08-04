package utils

import org.scalatest.WordSpec
import gp.imageprocessing.PMKUtils.SparseHistogramImpl

/**
 * Created by mjamroz on 03/08/14.
 */
class HistogramTests extends WordSpec{

  val dataSet = Array[Long](234,521,245,123,56,1,23,4,5,2,531,24,45)
  val anotherDs = Array(0.1,0.2,0.3,1.1,1.2,1.3,2.1,2.2,2.3)
  val validDs = Array(0.1,0.6,1.1,2.1,3.05,3.9,5.1)
  val justDs = Array(0.1,0.2,1.1,1.45,9.8,9.9,10.01,10.02)

  "Sparse histogram implementation" should {

	"count value occurrences properly" in {
	  val sparseHist = new SparseHistogramImpl(0.5,0,11)
	  justDs.foreach(value => sparseHist.updateBin(value))
	  assert(sparseHist.dim == 23)
	  assert(sparseHist.getNthDim(0) == 2)
	  assert(sparseHist.getNthDim(1) == 0)
	  assert(sparseHist.getNthDim(2) == 2)
	  assert(sparseHist.getNthDim(19) == 2)
	  assert(sparseHist.getNthDim(20) == 2)
	}
  }

}
