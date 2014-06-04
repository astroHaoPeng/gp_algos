package utils

import org.scalatest.{WordSpec}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import breeze.linalg._
import breeze.numerics.abs
import utils.KernelRequisites.{GaussianRbfParams, GaussianRbfKernel}
import utils.KernelRequisites.GaussianRbfParams
import utils.KernelRequisites.GaussianRbfKernel
import utils.NumericalUtils.Precision

/**
 * Created by mjamroz on 13/03/14.
 */

@RunWith(classOf[JUnitRunner])
class MatrixUtilsTest extends WordSpec {

  import MatrixUtils._

  val eps:Double = 0.001
  implicit val prec = Precision(p = eps)
  val lowerMatrix = DenseMatrix((0.3,0.,0.),(0.2,0.3,0.),(0.1,0.99,0.11))
  val upperMatrix = DenseMatrix((0.4,0.1,0.9),(0.,0.2,0.89),(0.,0.,.5))

  "forwardsolve function" should {

	"solve properly set of linear equations with triangular lower matrix" in {

	  val rhs = DenseVector(3., 2., 1.)
	  val solution: DenseVector[Double] = (lowerMatrix \ rhs)
	  val solutionFromMatrixUtils: DenseVector[Double] = MatrixUtils.forwardSolve(L = lowerMatrix, b = rhs)
	  assert(solution == solutionFromMatrixUtils)
	  compare2Vectors(solution,solutionFromMatrixUtils)
	}

	"solve properly set of linear equations with triangular upper matrix" in {

	  val rhs = DenseVector(7.,3.,4.)
	  val solution:DenseVector[Double] = (upperMatrix \ rhs)
	  val solutionFromMatrixUtils:DenseVector[Double] = MatrixUtils.backSolve(R = upperMatrix,b = rhs)
	  compare2Vectors(solution,solutionFromMatrixUtils)
	}

	"solve properly set of linear equations with triangular lower matrix and matrix on the right side" in {

	  val rhs = DenseMatrix((0.4,0.9),(0.8,0.3),(0.7,0.4))
	  val solution:DenseMatrix[Double] = (lowerMatrix \ rhs)
	  assert(solution.rows == 3 && solution.cols == 2)
	  val solutionFromMatrixUtils:DenseMatrix[Double] = MatrixUtils.forwardSolve(L = lowerMatrix,b = rhs)
	  assert(solutionFromMatrixUtils.rows == 3 && solutionFromMatrixUtils.cols == 2)
	  compare2Matrices(solution,solutionFromMatrixUtils)
	}

	"solve properly set of linear equations with triangular upper matrix and matrix on the right side" in {
	  val rhs = DenseMatrix((0.4,0.9),(0.8,0.3),(0.7,0.4))
	  val solution:DenseMatrix[Double] = (upperMatrix \ rhs)
	  assert(solution.rows == 3 && solution.cols == 2)
	  val solutionFromMatrixUtils:DenseMatrix[Double] = MatrixUtils.backSolve(R = upperMatrix,b = rhs)
	  assert(solutionFromMatrixUtils.rows == 3 && solutionFromMatrixUtils.cols == 2)
	  compare2Matrices(solution,solutionFromMatrixUtils)
	}

  }

  "vector by matrix elementwise multiplication" should {

	"work properly" in {
	  val vec = DenseVector(2.,3.)
	  val m = DenseMatrix((1.,2.),(4.,5.))
	  val resultMatrix:DenseMatrix[Double] = MatrixUtils.dvToElementWiseMultDenseVector(vec) :* m
	  assert(resultMatrix == DenseMatrix((2.,4.),(12.,15.)))
	  assert(resultMatrix == (m(::,*) :* vec))
	}

  }

  "division number by vector" should {

	"work properly" in {

	  val vec = DenseVector(1.,2.,4.)
	  val divided = (1 / vec)
	  assert(divided == DenseVector(1.,0.5,0.25))
	}

  }

  "kernel matrix building function" should {

	"work properly" in {

	  val input = DenseMatrix((2.4,1.3,1.9),(2.1,0.99,3.1),(1.89,2.01,4.))
	  val kernelFun = GaussianRbfKernel(GaussianRbfParams(signalVar = 1.,lengthScales = DenseVector(1.,1.,1.),
		noiseVar = 0.))
	  val kernelMatrix = MatrixUtils.buildKernelMatrix(kernelFun,input)
	  assert(kernelMatrix.rows == 3 && kernelMatrix.cols == 3)
	  (0 until 3).foreach {index => assert(kernelMatrix(index,index) == 1.)}
	  cholesky(kernelMatrix)
	}
  }

  "function inverting triangular matrix" in {
	val input = DenseMatrix((2.4,1.3,1.9),(2.1,0.99,3.1),(1.89,2.01,4.))
	val kernelFun = GaussianRbfKernel(GaussianRbfParams(signalVar = 1.,lengthScales = DenseVector(1.,1.,1.),
	  noiseVar = 0.))
	val kernelMatrix = MatrixUtils.buildKernelMatrix(kernelFun,input)
	val L:DenseMatrix[Double] = cholesky(kernelMatrix)
	val inversedK = inv(kernelMatrix)
	val inversedTriang = invTriangular(L,isUpper = false)
	val inversedKWithCholDecomp:DenseMatrix[Double] = inversedTriang.t * inversedTriang
	assert(inversedKWithCholDecomp ~= inversedK)
  }

  def compare2Vectors(vec1:DenseVector[Double],vec2:DenseVector[Double]):Unit = {
	(0 until vec1.length).foreach {indx => assert(abs(vec1(indx) - vec2(indx)) < eps)}
  }

  def compare2Matrices(m1:DenseMatrix[Double],m2:DenseMatrix[Double]):Unit = {
	(0 until m1.rows).foreach {rowIndx => compare2Vectors(m1(rowIndx,::).t, m2(rowIndx,::).t)}
  }

}
