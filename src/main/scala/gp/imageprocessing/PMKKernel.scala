package gp.imageprocessing

import utils.KernelRequisites.AbstractKernelFunc

/**
 * Created by mjamroz on 31/07/14.
 */
class PMKKernel(diameter:Double) extends AbstractKernelFunc[Array[Double]]{

  import PMKUtils._

  override def apply(obj1: Array[Double], obj2: Array[Double], sameIndex: Boolean): Double = {
	/*K(P,P) = K1(P,P)/sqrt(K1(P,P)*K1(P,P)) = 1*/
	if (sameIndex){return 1.}
	val unormalizedObj1Obj2:Double = unormalizedKernelVal(obj1,obj2,sameObj = false)
	val unormalizedObj1:Double = unormalizedKernelVal(obj1,obj1,sameObj = true)
	val unormalizedObj2:Double = unormalizedKernelVal(obj2,obj2,sameObj = true)
	(1/math.sqrt(unormalizedObj1*unormalizedObj2))*unormalizedObj1Obj2

  }

  private def unormalizedKernelVal(obj1:Array[Double],obj2:Array[Double],sameObj:Boolean):Double = {
	val (histsForObj1,histsForObj2) = sameObj match {
		case true =>
		  val extractedFeatures = featureExtractingFunc(obj1,diameter,sparseImpl = true)
		  (extractedFeatures,extractedFeatures)
		case false => (featureExtractingFunc(obj1,diameter,sparseImpl = true),featureExtractingFunc(obj2,diameter,sparseImpl = true))
	}
	require(histsForObj1.length == histsForObj2.length,
	  "Histogram vectors needs to have equal lengths. Ensure that they were built with the same diameter D")
	/*At level -1 */
	val firstPyramidIntersection = /*histogramIntersection(histsForObj1(0),histsForObj2(0))*/ 0
	val (kernelVal,_) = (1 until histsForObj1.length).foldLeft((0.,firstPyramidIntersection)){
	  case ((kernelVal,lastIntersectionVal),index) =>
		val weight = 1./(math.pow(2,index-1))
		val currIntersection:Int = histogramIntersection(histsForObj1(index),histsForObj2(index))
		(kernelVal + weight * (currIntersection - lastIntersectionVal), currIntersection)
	}
	kernelVal
  }

}
