package utils

import org.scalatest.WordSpec

/**
 * Created by mjamroz on 30/07/14.
 */
class SortingUtilsTest extends WordSpec{

  "Radix sort algorithm should sort the array and return it without affecting the input" in {

	val arr = Array(345,123,6,7,1231,6,5,2,90,987)
	val sortedArr = SortingUtils.radixSort(arr)
	assert(sortedArr === Array(2,5,6,6,7,90,123,345,987,1231))
	assert(arr != sortedArr)
	val arr1 = arr.clone()
	val sortedArr1 = SortingUtils.radixSort(arr1,immutableInput = false)
	assert(sortedArr1 === arr1)
  }

}
