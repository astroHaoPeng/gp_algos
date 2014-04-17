package gp.optimization

import org.scalatest.WordSpec
import utils.KernelRequisites.{GaussianRbfKernel, GaussianRbfParams}
import gp.regression.GpPredictor
import gp.optimization.GPOptimizer.GPOInput
import breeze.linalg.DenseVector
import utils.NumericalUtils.Precision

/**
 * Created by mjamroz on 18/04/14.
 */
class GPOptimizerTest extends WordSpec {

  import Optimization._

  implicit val precision = Precision(p = 0.0001)

  val quadraticFunction: objectiveFunction = {
	point => -2 * point(0) * point(0) + 6 * point(0) + 5
  }

  //TODO - single kernel configuration in all tests
  val (alpha, gamma, beta) = (1., 1., 0)
  //val (alpha,gamma,beta) = (1.,1.,sqrt(0.00125))
  //val (alpha,gamma,beta) = (164.52695987617062, 2.412787119003772, sqrt(0.00125))
  val defaultRbfParams: GaussianRbfParams = GaussianRbfParams(alpha = alpha, gamma = gamma, beta = beta)
  val gaussianKernel = GaussianRbfKernel(defaultRbfParams)
  val predictor = new GpPredictor(gaussianKernel)
  val gpoInput = GPOInput(ranges = IndexedSeq(-10 to 10), mParam = 10, cParam = 10, kParam = 0.5)
  val gpOptimizer = new GPOptimizer(predictor, noise = None)

  "GPOptimizer" should {

	"create a properly bounded point grid and evaluate it at the beginning" in {
	  val grid = gpOptimizer.prepareGrid(gpoInput.ranges)
	  assert(grid.rows == 3)
	  (0 until grid.rows).forall{
		index =>
		  val elem:Double = grid(index,::).toDenseVector(0)
		  (elem < gpoInput.ranges(0).end) && (elem > gpoInput.ranges(0).start)
	  }

	}

	"find the optimum of simple quadratic 1D function" in {
	  val (optimalSolution, optimalValue) = gpOptimizer.maximize(quadraticFunction, gpoInput)
	  println(s"optimal solution = ${DenseVector(optimalSolution)} , optimal value = ${optimalValue}")
	}

  }

}
