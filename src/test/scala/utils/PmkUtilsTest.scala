package utils

import org.scalatest.WordSpec
import gp.imageprocessing.{PMKKernel, PMKUtils, ImageProcessingUtils}
import gp.imageprocessing.ImageProcessingUtils.ImageLoadingSpec
import org.springframework.core.io.ClassPathResource

/**
 * Created by mjamroz on 31/07/14.
 */
class PmkUtilsTest extends WordSpec{

  val dataSet = Array[Long](234,521,245,123,56,1,23,4,5,2,531,24,45)
  val anotherDs = Array(0.1,0.2,0.3,1.1,1.2,1.3,2.1,2.2,2.3)
  val validDs = Array(0.1,0.6,1.1,2.1,3.05,3.9,5.1)

  "ImageProcessingUtils class" should {

	"compute prescaling output properly" in {
		val preScalingOut = ImageProcessingUtils.computePreScalingOut(dataSet,0.5)
	    assert(preScalingOut.maxValue == 531)
	  	assert(preScalingOut.minDistance == 1)
	  	assert(preScalingOut.scalingFactor == 0.5)
	}

  }

  "PmkUtils" should {

	"compute the feature extracting function properly" in {
	  val histoGrams = PMKUtils.featureExtractingFunc(anotherDs,3)
	  assert(histoGrams.length == 4)
	  val (hist1,hist2,hist3,hist4) = (histoGrams(0),histoGrams(1),histoGrams(2),histoGrams(3))
	  assert(hist1.dim == 6)
	  assert(hist1.bins === Array(3,0,3,0,3,0))
	  assert(hist2.dim == 3)
	  assert(hist2.bins === Array(3,3,3))
	  assert(hist3.dim == 2)
	  assert(hist3.bins === Array(6,3))
	  assert(hist4.dim == 1)
	  assert(hist4.bins === Array(9))
	}

	"compute the intersection between histograms properly" in {
	  val histograms = PMKUtils.featureExtractingFunc(anotherDs,3)
	  assert(histograms.length == 4)
	  val (hist1,hist2,hist3,hist4) = (histograms(0),histograms(1),histograms(2),histograms(3))
	  assert(PMKUtils.histogramIntersection(hist1,hist1) == 9)
	  assert(PMKUtils.histogramIntersection(hist2,hist2) == 9)
	  assert(PMKUtils.histogramIntersection(hist3,hist3) == 9)
	  assert(PMKUtils.histogramIntersection(hist4,hist4) == 9)
	}

	"compute all components properly to gain a proper PMKKernel value" in {
	  val kernel = new PMKKernel(5.5)
	  val kernelVal = kernel.apply(validDs,validDs,false)
	  assert(kernelVal > 0)

	}

	"repeat foregoing operations on real image" in {
	  val imageLoadSpec = ImageLoadingSpec(file = new ClassPathResource("images/emu.jpg").getFile,
		shiftByAlpha = true,convertToPositive = true)
	  val imageSpec = ImageProcessingUtils.loadImageInto2DArray(imageLoadSpec)
	  val imageSpec_ = imageSpec.right.get
	  val imageAs1Darr = ImageProcessingUtils.linearizeImage(imageSpec_.data)
	  val imageWithoutDupl = ImageProcessingUtils.removeDuplicates(imageAs1Darr)
	  val preScalingOut = ImageProcessingUtils.computePreScalingOut(imageWithoutDupl,0.5)
	  val preScaled1DImage = ImageProcessingUtils.preScaleInput(preScalingOut.scalingFactor,imageWithoutDupl)
	  val diam = preScalingOut.maxValue * preScalingOut.scalingFactor
	  val someHists = PMKUtils.featureExtractingFunc(preScaled1DImage,diam,sparseImpl = true)
	  assert(!someHists.isEmpty)

	  val kernel = new PMKKernel(diam)
	  val kernelVal = kernel.apply(preScaled1DImage,preScaled1DImage,false)
	  assert(kernelVal > 0)
	}



  }

}
