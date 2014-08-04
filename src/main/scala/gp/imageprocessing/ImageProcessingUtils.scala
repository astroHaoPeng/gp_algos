package gp.imageprocessing

import utils.SortingUtils
import java.io.File
import javax.imageio.ImageIO
import java.awt.image.BufferedImage

/**
 * Created by mjamroz on 30/07/14.
 */
object ImageProcessingUtils {

  case class PreScalingOutput(maxValue:Long,scalingFactor:Double,minDistance:Long)
  case class ImageSpec(data:Array[Array[Long]],height:Int,width:Int)
  case class ImageLoadingSpec(file:File,convertToPositive:Boolean,shiftByAlpha:Boolean)

  def linearizeImage(image2D:Array[Array[Long]]):Array[Long] = {
	image2D.flatMap(arr1d => arr1d)
  }

  def computePreScalingOut(image1D:Array[Long],minDistance:Double):PreScalingOutput = {

	val sortedPixels = SortingUtils.radixSort(image1D)
	val minDistanceBetweenInts = (1 until sortedPixels.length).foldLeft(Long.MaxValue){
	  case (minCurrDist,index) =>
		val dist = math.abs(sortedPixels(index) - sortedPixels(index-1))
	  	val newMinDist = if (dist < minCurrDist && dist != 0){
			dist
		} else {minCurrDist}
		newMinDist
	}
	val scalingFactor:Double = minDistance / minDistanceBetweenInts
	PreScalingOutput(maxValue = sortedPixels.last,scalingFactor = scalingFactor,minDistance = minDistanceBetweenInts)
  }

  def preScaleInput(factor:Double,image1D:Array[Long]):Array[Double] = {
	image1D.map {elem =>
	  elem * factor
	}
  }

  def loadImageInto2DArray(imageLoadingSpec:ImageLoadingSpec):Either[Exception,ImageSpec] = {
	var imgBuff:BufferedImage = null
	try{
	  val (file,convertToPos,shiftByAlpha) =
		(imageLoadingSpec.file,imageLoadingSpec.convertToPositive,imageLoadingSpec.shiftByAlpha)
	  imgBuff = ImageIO.read(file)
	  val result = Array.ofDim[Long](imgBuff.getHeight,imgBuff.getWidth)
	  for (row <- 0 until imgBuff.getHeight){
		for (col <- 0 until imgBuff.getWidth){

		  val pixelVal:Int = imgBuff.getRGB(col,row)
		  val finalValue:Long = convertToPos  match {
			case true if shiftByAlpha => (pixelVal & 0xffffffffl) - (0xffffffffl & 0xff000000l)
			case false if shiftByAlpha => ???
			case true => (pixelVal & 0xffffffffl)
			case false => pixelVal.toLong
		  }

		  result(row)(col) = finalValue
		}
	  }
	  Right(ImageSpec(data = result,height = imgBuff.getHeight,width = imgBuff.getWidth))
	} catch {
	  case e:Exception => Left(e)
	}
  }

  def removeDuplicates(image1D:Array[Long]):Array[Long] = {
	image1D.toSet.toArray
  }
  /*type imageConvertingFunc = BufferedImage => BufferedImage
  val removeAlphaCompFunc:imageConvertingFunc = {
	buffImage:BufferedImage =>
	  for (row <- 0 until buffImage.getHeight){
		for (col <- 0 until buffImage.getWidth){
		  val rgbWithAlpha = buffImage.getRGB(col,row)
		  val currColor = new Color(rgbWithAlpha)
		  val colorWithoutAlpha = new Color(currColor.getRed,currColor.getGreen,currColor.getBlue,0)
		  buffImage.setRGB(col,row,colorWithoutAlpha.getRGB)
		}
	  }
	  buffImage
  }
  val convertingFuncs:Seq[imageConvertingFunc] = Seq(removeAlphaCompFunc)
  def applyConversion(initBuff:BufferedImage):BufferedImage = {
	convertingFuncs.foldLeft(initBuff){
	  case (currBuff,convertingFunc) =>
	  	convertingFunc(currBuff)
	}
  } */

}
