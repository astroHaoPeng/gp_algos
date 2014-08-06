package gp.imageprocessing

import utils.SortingUtils
import java.io.{FileFilter, File}
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import org.springframework.core.io.ClassPathResource

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

  def loadAndConvertImage(file:File):(Array[Double],PreScalingOutput) = {
	val imageLoadSpec = ImageLoadingSpec(file = file,
	  shiftByAlpha = true,convertToPositive = true)
	val imageSpec = ImageProcessingUtils.loadImageInto2DArray(imageLoadSpec)
	val imageSpec_ = imageSpec.right.get
	val imageAs1Darr = ImageProcessingUtils.linearizeImage(imageSpec_.data)
	val imageWithoutDupl = ImageProcessingUtils.removeDuplicates(imageAs1Darr)
	val preScalingOut = ImageProcessingUtils.computePreScalingOut(imageWithoutDupl,0.5)
	(ImageProcessingUtils.preScaleInput(preScalingOut.scalingFactor,imageWithoutDupl),preScalingOut)
  }

  def loadAndConvertImage(fileName:String,resourceBaseDir:Boolean=true):(Array[Double],PreScalingOutput) = {
	val file = if (resourceBaseDir){new ClassPathResource(fileName).getFile}
	else {new File(fileName)}
	loadAndConvertImage(file)
  }

  def loadAndConvertImagesFromDir(dirName:String):IndexedSeq[(Array[Double],PreScalingOutput)] = {
	val file = new File(dirName)
	file.listFiles.filterNot(_.isDirectory).map{ imageFile =>
	  	assert(imageFile.exists())
	    loadAndConvertImage(imageFile)
	}
  }

}
