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

  val ls:DenseVector[Double] = DenseVector.ones[Double](5)

  "kernel hyperparams" should {

	"return proper values when it's converted to dense vector" in {
	  val hyperParams = GaussianRbfParams(signalVar = 1.,lengthScales = ls,noiseVar = 0.)
	  assert(hyperParams.toDenseVector == DenseVector(1.,1.,1.,1.,1.,1.,0.))
	}

	"return proper value at specified position" in {
	  val hyperParams = GaussianRbfParams(signalVar = 1.,lengthScales = DenseVector(5.,2.,3.),noiseVar = 0.)
	  assert(hyperParams.getAtPosition(1) == 1.)
	  assert(hyperParams.getAtPosition(2) == 5.)
	  assert(hyperParams.getAtPosition(3) == 2.)
	  assert(hyperParams.getAtPosition(4) == 3.)
	  assert(hyperParams.getAtPosition(5) == 0.)
	  intercept[MatchError]{
		hyperParams.getAtPosition(6)
	  }
	}
  }

  "kernel function" should {

	"update it's params"  in {

	  val initKernelFunc = GaussianRbfKernel(GaussianRbfParams(signalVar = 1.,lengthScales = ls,noiseVar = 0.))
	  assert(initKernelFunc.rbfParams == GaussianRbfParams(signalVar = 1.,lengthScales = ls,noiseVar = 0.))
	  val updatedKernelFunc = initKernelFunc.changeHyperParams(DenseVector(2.,3.,3.,3.,3.,3.,0.))
	  assert(updatedKernelFunc.rbfParams == GaussianRbfParams(signalVar = 2.,lengthScales = ls :* 3.,noiseVar = 0.))

	}

  }

}
