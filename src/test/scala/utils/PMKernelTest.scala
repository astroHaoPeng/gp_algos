package utils

import org.scalatest.WordSpec
import gp.imageprocessing.PMKKernel

/**
 * Created by mjamroz on 03/08/14.
 */
class PMKernelTest extends WordSpec with PmkTestingTrait{

  "PMKernel" should {

	"compute similarity measures for sample images" in {
		val (emuImage,emuScalingOut) = loadAndConvertImage("images/emu.jpg")
	  	val (dolImage,dolScalingOut) = loadAndConvertImage("images/dolphin.jpg")
	  	val (emu1Image,emu1ScalingOut) = loadAndConvertImage("images/emu1.jpg")
	  	val maxDiam = Seq(emuScalingOut.scalingFactor*emuScalingOut.maxValue,
		  dolScalingOut.scalingFactor*dolScalingOut.maxValue,emu1ScalingOut.scalingFactor*emu1ScalingOut.maxValue).max
	  	val pmKernel = new PMKKernel(maxDiam)
	  	val emuDolKernelVal = pmKernel.apply(emuImage,dolImage,sameIndex = false)
	  	val emuEmu1KernelVal = pmKernel.apply(emuImage,emu1Image,sameIndex = false)
	    println(s"Emu - Dolphin Val = $emuDolKernelVal")
	  	println(s"Emu1 - Emu Val = $emuEmu1KernelVal")
	}
  }

}
