package utils

import org.scalatest.WordSpec
import gp.imageprocessing.ImageProcessingUtils
import org.springframework.core.io.ClassPathResource
import gp.imageprocessing.ImageProcessingUtils.ImageLoadingSpec

/**
 * Created by mjamroz on 02/08/14.
 */
class ImageProcessingUtilsTest extends WordSpec{

  "ImageProcessingUtils class" should {

	"load test image properly" in {

	  val imageLoadSpec = ImageLoadingSpec(file = new ClassPathResource("images/emu.jpg").getFile,
	  shiftByAlpha = true,convertToPositive = true)
	  val imageSpec = ImageProcessingUtils.loadImageInto2DArray(imageLoadSpec)
	  assert(imageSpec.isRight)
	  val imageSpec_ = imageSpec.right.get
	  assert(ImageProcessingUtils.linearizeImage(imageSpec_.data).forall(_ >= 0))
	  assert(ImageProcessingUtils.linearizeImage(imageSpec_.data).forall(_ <= math.pow(2,24)-1))
	}

	"prescale the image properly" in {

	  val imageLoadSpec = ImageLoadingSpec(file = new ClassPathResource("images/emu.jpg").getFile,
		shiftByAlpha = true,convertToPositive = true)
	  val imageSpec = ImageProcessingUtils.loadImageInto2DArray(imageLoadSpec)
	  assert(imageSpec.isRight)
	  val imageSpec_ = imageSpec.right.get
	  val imageAs1Darr = ImageProcessingUtils.linearizeImage(imageSpec_.data)
	  assert(imageAs1Darr.length == imageSpec_.height*imageSpec_.width)
	  val scalingOut = ImageProcessingUtils.computePreScalingOut(imageAs1Darr,0.5)
	  assert(scalingOut.scalingFactor > 0.)
	  val image1DWithoutDupl = ImageProcessingUtils.removeDuplicates(imageAs1Darr)
	  val scalingOut1 = ImageProcessingUtils.computePreScalingOut(image1DWithoutDupl,0.5)
	  assert(scalingOut1.scalingFactor > 0.)
	}

  }

}
