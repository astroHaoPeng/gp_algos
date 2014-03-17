package utils

import breeze.linalg.{DenseVector, DenseMatrix}

/**
 * Created by mjamroz on 11/03/14.
 */
object MatrixUtils {

  import KernelRequisites._

  type rowMatrixRange = Int => Range

  /* Set of linear equations */
  def forwardSolve(L:DenseMatrix[Double],b:DenseVector[Double]):DenseVector[Double] = {
	val solutionRange = (0 until L.rows)
	val rowRange:rowMatrixRange = (0 until _)
	solveTriangular(L,b,solutionRange,rowRange)
  }

  def backSolve(R:DenseMatrix[Double],b:DenseVector[Double]):DenseVector[Double] = {
  	val solutionRange = ((R.rows-1).to(0,-1))
	val rowRange:rowMatrixRange = (R.cols-1).until(_,-1)
	solveTriangular(R,b,solutionRange,rowRange)
  }

  def forwardSolve(L:DenseMatrix[Double],b:DenseMatrix[Double]):DenseMatrix[Double] = {
	solveTriangular(L,b,forwardSolve(_,_))
  }

  def backSolve(R:DenseMatrix[Double],b:DenseMatrix[Double]):DenseMatrix[Double] = {
	solveTriangular(R,b,backSolve(_,_))
  }

  def cloneCols(vec:DenseVector[Double],colNum:Int):DenseMatrix[Double] = {
	(0 until colNum).foldLeft(DenseMatrix.zeros[Double](vec.length,colNum)){
	  case (result,colIndex) =>
	  	result(::,colIndex) := vec; result
	}
  }

  def buildKernelMatrix(kernelFun:KernelFunc,input1:DenseMatrix[Double],input2:DenseMatrix[Double]):kernelMatrixType = {
	val (rowSize,colSize) = (input1.rows,input2.rows)
	val result:kernelMatrixType = DenseMatrix.zeros[Double](rowSize,colSize)
	for (i <- 0.until(rowSize)){
	  for (j <- 0.until(colSize)){
		val value = kernelFun(input1(i,::).toDenseVector,input2(j,::).toDenseVector)
		result.update(i,j,value)
	  }
	}
	result
  }

  def buildKernelMatrix(kernelFun:KernelFunc,data:DenseMatrix[Double])
  :kernelMatrixType = {
	val rowSize:Int = data.rows
	val result:kernelMatrixType = DenseMatrix.zeros[Double](rowSize,rowSize)
	for (i <- 0.until(rowSize)){
	  for (j <- 0.to(i)){
		val value = kernelFun(data(i,::).toDenseVector,data(j,::).toDenseVector)
		/*Assumption that kernel matrix is symmetric*/
		result.update(i,j,value)
		result.update(j,i,value)
	  }
	}
	result
  }

  def buildKernelMatrix(data:DenseMatrix[Double])(f:(DenseVector[Double],DenseVector[Double]) => Double):kernelMatrixType = {
	val rowSize:Int = data.rows
	val result:kernelMatrixType = DenseMatrix.zeros[Double](rowSize,rowSize)
	for (i <- 0.until(rowSize)){
	  for (j <- 0.to(i)){
		val value = f(data(i,::).toDenseVector,data(j,::).toDenseVector)
		/*Assumption that kernel matrix is symmetric*/
		result.update(i,j,value)
		result.update(j,i,value)
	  }
	}
	result
  }

  private def solveTriangular(L:DenseMatrix[Double],b:DenseMatrix[Double],
							  solvingFun:(DenseMatrix[Double],DenseVector[Double]) => DenseVector[Double]):DenseMatrix[Double] = {
	(0 until b.cols).foldLeft(DenseMatrix.zeros[Double](L.rows,b.cols)){
	  case (xs,rhsIndex) =>
		xs(::,rhsIndex) := solvingFun(L,b(::,rhsIndex)); xs
	}
  }

  private def solveTriangular(L:DenseMatrix[Double],b:DenseVector[Double],solutionRange:Range,
							  matrixRowRange:rowMatrixRange):DenseVector[Double] = {
	require(L.rows == L.cols)
	val n = L.rows
	solutionRange.foldLeft(DenseVector.zeros[Double](n)){case (x,rowIndex) =>
	  val sumOfSolutionsExceptXRowIndex = matrixRowRange(rowIndex).foldLeft(0.){case (acc,colIndex) =>
		acc + L(rowIndex,colIndex)*x(colIndex)
	  }
	  x.update(rowIndex,(b(rowIndex) - sumOfSolutionsExceptXRowIndex)/L(rowIndex,rowIndex)); x
	}
  }

  class IntDividingVector(i:Int) {
	def / (vec:DenseVector[Double]):DenseVector[Double] = {
	  (0 until vec.length).foldLeft(DenseVector.zeros[Double](vec.length)){
		case (resultVec,index) => resultVec.update(index,i/vec(index)); resultVec
	  }
	}
  }

  class ElementWiseMultDenseVector(val dv:DenseVector[Double]) {

	def :* (m:DenseMatrix[Double]):DenseMatrix[Double] = {
	  0.until(m.rows).foldLeft(DenseMatrix.zeros[Double](m.rows,m.cols)){
		case (result,rowIndex) =>
		  result(rowIndex,::) := m :* dv(rowIndex); result
	  }
	}

  }

  implicit def intToIntDividingVector(i:Int):IntDividingVector = new IntDividingVector(i)

  implicit def dvToElementWiseMultDenseVector(dv:DenseVector[Double]):ElementWiseMultDenseVector =
	new ElementWiseMultDenseVector(dv)
}
