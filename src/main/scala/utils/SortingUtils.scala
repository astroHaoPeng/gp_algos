package utils

/**
 * Created by mjamroz on 30/07/14.
 */
object SortingUtils {

  import collection._

  def radixSort(input:Array[Int]) = {
	val RADIX = 10
	// declare and initialize bucket[]
	val bucket:Array[mutable.ListBuffer[Int]] = new Array[mutable.ListBuffer[Int]](10)
	for (i <- 0 until bucket.length){
	  bucket(i) = mutable.ListBuffer.empty[Int]
	}
	val sortedResult = input.clone()

	// sort
	var (maxLength,tmp,placement) = (false,-1,1)

	while (!maxLength) {
	  maxLength = true;
	  // split input between lists
	  sortedResult.foreach {i =>
		tmp = i / placement
		bucket(tmp % RADIX) += i
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
