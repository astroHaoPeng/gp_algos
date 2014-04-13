package utils

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Created by mjamroz on 14/03/14.
 */
object KernelRequisites {

  import scala.math._

  type featureVector = DenseVector[Double]
  type kernelMatrixType = DenseMatrix[Double]

  trait KernelFuncHyperParams{

	def getAtPosition(i:Int):Double
	def toDenseVector:DenseVector[Double]
	def fromDenseVector(dv:DenseVector[Double]):KernelFuncHyperParams
  }

  trait KernelFunc{

	def apply(obj1:featureVector,obj2:featureVector,sameIndex:Boolean):Double
	def hyperParametersNum:Int
	def derAfterHyperParam(paramNum:Int):(featureVector,featureVector,Boolean) => Double
	def changeHyperParams(dv:DenseVector[Double]):KernelFunc
  }


  case class GaussianRbfParams(alpha:Double,gamma:Double,beta:Double) extends KernelFuncHyperParams{
	def getAtPosition(i: Int): Double = {
	  i match {
		case 1 => alpha
		case 2 => gamma
		case 3 => beta
	  }
	}

	override def toDenseVector: DenseVector[Double] = {
	  DenseVector(alpha,gamma,beta)
	}

	override def fromDenseVector(dv: DenseVector[Double]): GaussianRbfParams = {
	  val (newAlpha,newGamma,newBeta) = dv.length match {
		case 0 => (alpha,gamma,beta)
		case 1 => (dv(0),gamma,beta)
		case 2 => (dv(0),dv(1),beta)
		case 3 => (dv(0),dv(1),dv(2))
	  }
	  this.copy(alpha = newAlpha,gamma = newGamma,beta = newBeta)
	}
  }
  //function of form k(x_p,x_q) = alpha*exp(-0.5*(gamma^2)*t(x_p-x_q)*(x_p-x_q)) + (beta^2)*(p == q)
  case class GaussianRbfKernel(rbfParams:GaussianRbfParams) extends KernelFunc {

	private val (alpha,gamma,beta) = (rbfParams.alpha,rbfParams.gamma,rbfParams.beta)

  	def apply(obj1:featureVector,obj2:featureVector,sameIndex:Boolean):Double = {
	  val diff = (obj1 - obj2)
	  val valWithoutNoise:Double = alpha*exp(-0.5*gamma*gamma*(diff dot diff))
	  val retValue = if (!sameIndex){valWithoutNoise} else {
		valWithoutNoise + beta*beta
	  }
	  retValue
	}

	def hyperParametersNum: Int = 3

	def derAfterHyperParam(paramNum: Int):
		(KernelRequisites.featureVector, KernelRequisites.featureVector,Boolean) => Double = {
	  		case (vec1,vec2,sameIndex) =>
		val diff = (vec1 - vec2)
		val prodOfDiffs = diff dot diff
	    paramNum match {
		  case 1 => exp(-0.5*gamma*gamma*prodOfDiffs)
		  case 2 => alpha*exp(-0.5*gamma*gamma*prodOfDiffs)*(-1.)*gamma*prodOfDiffs
		  case 3 => if (sameIndex){2*beta} else {0.}
		}
	}

	def changeHyperParams(dv: DenseVector[Double]): GaussianRbfKernel = {
	  val newHyperParams = rbfParams.fromDenseVector(dv)
	  GaussianRbfKernel(newHyperParams)
	}
  }

  def testTrainKernelMatrix[T](test:DenseMatrix[Double],train:DenseMatrix[Double],kernelFun:KernelFunc):kernelMatrixType = {
	val result:kernelMatrixType = DenseMatrix.zeros[Double](test.rows,train.rows)
	for (i <- 0.until(test.rows)){
	  for (j <- 0.until(train.rows)){
		result(i until (i+1),j until (j+1)) := kernelFun(test(i,::).toDenseVector,train(j,::).toDenseVector,false)
	  }
	}
	result
  }

}
