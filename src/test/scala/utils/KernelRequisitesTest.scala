package utils

import org.scalatest.WordSpec
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import utils.KernelRequisites.{GaussianRbfParams, GaussianRbfKernel}
import breeze.linalg.DenseVector

/**
 * Created by mjamroz on 18/03/14.
 */

@RunWith(classOf[JUnitRunner])
class KernelRequisitesTest extends WordSpec {

  "kernel function" should {

	"update it's params"  in {

	  val initKernelFunc = GaussianRbfKernel(GaussianRbfParams(alpha = 1.,gamma = 2.5,beta = 0.))
	  assert(initKernelFunc.rbfParams == GaussianRbfParams(alpha = 1.,gamma = 2.5,beta = 0.))
	  val updatedKernelFunc = initKernelFunc.changeHyperParams(DenseVector(2.,3.,0.)).asInstanceOf[GaussianRbfKernel]
	  assert(updatedKernelFunc.rbfParams == GaussianRbfParams(alpha = 2.,gamma = 3.,beta = 0.))

	}

  }

}
