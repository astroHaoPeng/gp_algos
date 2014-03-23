package utils

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Created by mjamroz on 14/03/14.
 */
object KernelRequisites {

  import scala.math._

  type featureVector = DenseVector[Double]
  type kernelMatrixType = DenseMatrix[Double]

  trait KernelFunc{
	def apply(obj1:featureVector,obj2:featureVector):Double
	def hyperParametersNum:Int
	def derAfterHyperParam(paramNum:Int):(featureVector,featureVector) => Double
	def changeHyperParams(dv:DenseVector[Double]):KernelFunc
  }

  trait KernelFuncHyperParams {
	def getAtPosition(i:Int):Double
	def toDenseVector:DenseVector[Double]
	def fromDenseVector(dv:DenseVector[Double]):KernelFuncHyperParams
  }

  case class GaussianRbfParams(alpha:Double,gamma:Double) extends KernelFuncHyperParams{
	def getAtPosition(i: Int): Double = {
	  i match {
		case 1 => alpha
		case 2 => gamma
	  }
	}

	def toDenseVector: DenseVector[Double] = {
	  DenseVector(alpha,gamma)
	}

	def fromDenseVector(dv: DenseVector[Double]): KernelFuncHyperParams = {
	  GaussianRbfParams(alpha = dv(0),gamma = dv(1))
	}
  }
  //function of form k(x,y) = alpha*exp(-0.5*(gamma^2)*t(x-y)*(x-y))
  case class GaussianRbfKernel(rbfParams:GaussianRbfParams) extends KernelFunc{

	private val (alpha,gamma) = (rbfParams.alpha,rbfParams.gamma)

  	def apply(obj1:featureVector,obj2:featureVector):Double = {
	  val diff = (obj1 - obj2)
	  val retValue = alpha*exp(-0.5*gamma*gamma*(diff dot diff))
	  retValue
	}

	def hyperParametersNum: Int = 2

	def derAfterHyperParam(paramNum: Int):
		(KernelRequisites.featureVector, KernelRequisites.featureVector) => Double = {
	  		case (vec1,vec2) =>
		val diff = (vec1 - vec2)
		val prodOfDiffs = diff dot diff
	    paramNum match {
		  case 1 => exp(-0.5*gamma*gamma*prodOfDiffs)
		  case 2 => alpha*exp(-0.5*gamma*prodOfDiffs)*(-1.)*gamma*prodOfDiffs
		}
	}

	def changeHyperParams(dv: DenseVector[Double]): KernelFunc = {
	  GaussianRbfKernel(GaussianRbfParams(alpha = dv(0),gamma = dv(1)))
	}
  }

  def buildKernelMatrix(kernelFun:KernelFunc,data:DenseMatrix[Double],beta:Double=3.45)
  					:kernelMatrixType = {
	val rowSize:Int = data.rows
	val result:kernelMatrixType = DenseMatrix.zeros[Double](rowSize,rowSize)
	for (i <- 0.until(rowSize)){
	  for (j <- 0.to(i)){
		val value = (i == j) match {
		  case false => kernelFun(data(i,::).toDenseVector,data(j,::).toDenseVector)
		  case true => kernelFun(data(i,::).toDenseVector,data(j,::).toDenseVector) + 1/beta
		}
		result(i until (i+1),j until (j+1)) := value
		result(j until (j+1),i until (i+1)) := value
	  }
	}
	result
  }

  def testTrainKernelMatrix(test:DenseMatrix[Double],train:DenseMatrix[Double],kernelFun:KernelFunc):kernelMatrixType = {
	val result:kernelMatrixType = DenseMatrix.zeros[Double](test.rows,train.rows)
	for (i <- 0.until(test.rows)){
	  for (j <- 0.until(train.rows)){
		result(i until (i+1),j until (j+1)) := kernelFun(test(i,::).toDenseVector,train(j,::).toDenseVector)
	  }
	}
	result
  }

}
