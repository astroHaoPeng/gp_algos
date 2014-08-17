package utils

import org.scalatest.WordSpec

/**
 * Created by mjamroz on 30/07/14.
 */
class SortingUtilsTest extends WordSpec{

  "Radix sort algorithm should sort the array and return it without affecting the input" in {

	val arr = Array(3453131231l,1231231231l,6123313l,7l,1231l,6l,5l,2l,90l,987l)
	val sortedArr = SortingUtils.radixSort(arr)
	assert(sortedArr === arr.sortWith(_ < _))
	assert(arr != sortedArr)
	val arr1 = arr.clone()
	val sortedArr1 = SortingUtils.radixSort(arr1,immutableInput = false)
	assert(sortedArr1 === arr1)
  }

}
