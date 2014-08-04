package utils

import gp.imageprocessing.ImageProcessingUtils.{PreScalingOutput, ImageLoadingSpec}
import org.springframework.core.io.ClassPathResource
import gp.imageprocessing.ImageProcessingUtils

/**
 * Created by mjamroz on 03/08/14.
 */
trait PmkTestingTrait {

  def loadAndConvertImage(fileName:String):(Array[Double],PreScalingOutput) = {
	val imageLoadSpec = ImageLoadingSpec(file = new ClassPathResource(fileName).getFile,
	  shiftByAlpha = true,convertToPositive = true)
	val imageSpec = ImageProcessingUtils.loadImageInto2DArray(imageLoadSpec)
	val imageSpec_ = imageSpec.right.get
	val imageAs1Darr = ImageProcessingUtils.linearizeImage(imageSpec_.data)
	val imageWithoutDupl = ImageProcessingUtils.removeDuplicates(imageAs1Darr)
	val preScalingOut = ImageProcessingUtils.computePreScalingOut(imageWithoutDupl,0.5)
	(ImageProcessingUtils.preScaleInput(preScalingOut.scalingFactor,imageWithoutDupl),preScalingOut)
  }

}
