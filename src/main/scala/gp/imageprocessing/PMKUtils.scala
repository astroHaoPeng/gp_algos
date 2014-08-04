package gp.imageprocessing

/**
 * Created by mjamroz on 30/07/14.
 */
object PMKUtils {

  trait Histogram {

	val bins:Array[Int]

	def updateBin(value:Double)

	def dim:Int

	def getNthDim(n:Int):Int

  }

  class DefaultHistogramImpl(sideLength:Double,leftBound:Double,rightBound:Double) extends Histogram{

	require(leftBound < rightBound,"Left boundary ought to be lesser than the right one")
	val bins:Array[Int] = allocBins

	def updateBin(value:Double) = {
	  val indx = (value / sideLength).toInt
	  require(indx < bins.length,s"Value $indx needs to be smaller than ${bins.length}")
	  bins(indx) += 1
	}

	def dim = bins.length

	def getNthDim(n:Int):Int = {
	  require(n >= 0 && n < bins.length,s"Dimension needs to lie between 0 and ${bins.length}")
	  bins(n)
	}

	def getBins = bins

	private def allocBins = {
		val intervalLength:Double = rightBound - leftBound
	    val arrLength = math.ceil(intervalLength / sideLength).toInt+1
	  	Array.fill[Int](arrLength)(0)
	}

  }

  class SparseHistogramImpl(sideLength:Double,leftBound:Double,rightBound:Double) extends Histogram {

	import collection._
	private val binCounts:mutable.Map[Double,Int] = mutable.Map.empty[Double,Int]

	override def getNthDim(n: Int): Int = {
		val leftLimit = sideLength*n
	  	binCounts.get(leftLimit).fold(0){count => count}
	}

	override def dim: Int = {
	  val intervalLength:Double = rightBound - leftBound
	  math.ceil(intervalLength / sideLength).toInt+1
	}

	override def updateBin(value: Double): Unit = {
	  val k = math.floor(value / sideLength).toInt
	  val leftLimit = k*sideLength
	  binCounts.get(leftLimit) match {
		case Some(count) => binCounts.put(leftLimit,count+1)
		case None => binCounts.put(leftLimit,1)  
	  }
	}

	override val bins: Array[Int] = Array()
  }

  def featureExtractingFunc(image1D:Array[Double],diameter:Double):IndexedSeq[Histogram] = {
	featureExtractingFunc(image1D,diameter,true)
  }

  val histFactory = {(sparseImpl:Boolean,sideLength:Double,diameter:Double) =>
	if (sparseImpl){
	  new SparseHistogramImpl(sideLength,0,diameter)
	} else {new DefaultHistogramImpl(sideLength,0,diameter)}
  }

  def featureExtractingFunc(image1D:Array[Double],diameter:Double,sparseImpl:Boolean):IndexedSeq[Histogram] = {
	val L:Int = math.ceil(math.log(diameter) / math.log(2)).toInt
	(0 to (L+1)).foldLeft(IndexedSeq.empty[Histogram]){
	  case (seqOfHists,l) =>
		val histSideLength:Double = math.pow(2,l-1)
		val histogramForLevelL = histFactory(sparseImpl,histSideLength,diameter)
	    image1D.foreach { preScaledPixel =>
			histogramForLevelL.updateBin(preScaledPixel)
		}
	  	seqOfHists :+ histogramForLevelL
	}
  }

  def histogramIntersection(hist1:Histogram,hist2:Histogram):Int = {
	require(hist1.dim == hist2.dim)
	(0 until hist1.dim).foldLeft(0){
	  case (currValue,index) =>
	  	currValue + math.min(hist1.getNthDim(index),hist2.getNthDim(index))
	}
  }


}
