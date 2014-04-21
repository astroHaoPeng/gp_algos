package gp.optimization

import org.scalatest.WordSpec
import utils.KernelRequisites.{GaussianRbfKernel, GaussianRbfParams}
import gp.regression.GpPredictor
import gp.optimization.GPOptimizer.GPOInput
import breeze.linalg.DenseVector
import utils.NumericalUtils.Precision
import breeze.numerics.sqrt
import org.junit.runner.RunWith
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner
import org.springframework.test.context.{TestContextManager, ContextConfiguration}
import org.springframework.beans.factory.annotation.Autowired

/**
 * Created by mjamroz on 18/04/14.
 */

@RunWith(classOf[SpringJUnit4ClassRunner])
@ContextConfiguration(locations = Array("classpath:config/spring-context.xml"))
class GPOptimizerTest extends WordSpec {

  import OptimizationUtils.Functions1D._
  import OptimizationUtils.Functions2D._

  implicit val precision = Precision(p = 0.0001)

  new TestContextManager(this.getClass).prepareTestInstance(this)

  @Autowired
  var gpOptimizer:GPOptimizer = _

  "GPOptimizer" should {

	"create a properly bounded point grid and evaluate it at the beginning" in {
	  val gpoInput = GPOInput(ranges = IndexedSeq(-6 to 6), mParam = 50, cParam = 5, kParam = 2.)
	  val grid = gpOptimizer.prepareGrid(gpoInput.ranges)
	  assert(grid.rows == 3)
	  (0 until grid.rows).forall{
		index =>
		  val elem:Double = grid(index,::).toDenseVector(0)
		  (elem < gpoInput.ranges(0).end) && (elem > gpoInput.ranges(0).start)
	  }

	}

	"find the optimum of simple quadratic 1D function" in {
	  val gpoInput = GPOInput(ranges = IndexedSeq(-6 to 6), mParam = 50, cParam = 5, kParam = 2.)
	  val (optimalSolution, optimalValue) = gpOptimizer.minimize(strangeFunc1, gpoInput)
	  println(s"optimal solution = ${DenseVector(optimalSolution)} , optimal value = ${optimalValue}")
	}

	"find the optimum of rastrigin 2D function" in {
	  val gpoInput = GPOInput(ranges = IndexedSeq(-5 to 5,-5 to 5), mParam = 100, cParam = 10, kParam = 2.)
	  val (optimalSolution, optimalValue) = gpOptimizer.minimize(rastriginFunc, gpoInput)
	  println(s"Rastr - optimal solution = ${DenseVector(optimalSolution)} , optimal value = ${optimalValue}")
	}

  }

}
