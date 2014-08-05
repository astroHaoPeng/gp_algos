package utils

/**
 * Created by mjamroz on 30/07/14.
 */
object SortingUtils {

  import collection._

  type sortedType = Long

  def radixSort(input:Array[Long]):Array[Long] = {
	radixSort(input,true)
  }

  def radixSort(input:Array[sortedType],immutableInput:Boolean) = {
	val RADIX = 10
	// declare and initialize bucket[]
	val bucket:Array[mutable.ListBuffer[sortedType]] = new Array[mutable.ListBuffer[sortedType]](10)
	for (i <- 0 until bucket.length){
	  bucket(i) = mutable.ListBuffer.empty[sortedType]
	}
	val sortedResult = if (immutableInput){
	  input.clone()
	} else {
	  input
	}
	// sort
	var (maxLength,tmp,placement) = (false,-1l,1l)

	while (!maxLength) {
	  maxLength = true;
	  // split input between lists
	  sortedResult.foreach {i =>
		tmp = i / placement
		bucket((tmp % RADIX).toInt) += i
		if (maxLength && tmp > 0){
		  maxLength = false
		}
	  }
	  var a = 0
	  for (b <- 0 until RADIX){
		bucket(b).foreach { i =>
			sortedResult(a) = i
		  	a += 1
		}
		bucket(b).clear()
	  }
	  placement *= RADIX

	}
	sortedResult
  }

}
