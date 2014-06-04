package utils

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Created by mjamroz on 14/03/14.
 */
object KernelRequisites {

  import scala.math._

  type featureVector = DenseVector[Double]
  type kernelMatrixType = DenseMatrix[Double]
  type kernelDerivative = (featureVector,featureVector) => DenseVector[Double]

  trait KernelFuncHyperParams{
	/*Indexed from 1, not from 0 !!!*/
	def getAtPosition(i:Int):Double
	def toDenseVector:DenseVector[Double]
	def fromDenseVector(dv:DenseVector[Double]):KernelFuncHyperParams
  }

  trait KernelFunc{

	def apply(obj1:featureVector,obj2:featureVector,sameIndex:Boolean):Double
	def hyperParametersNum:Int
	def derAfterHyperParam(paramNum:Int):(featureVector,featureVector,Boolean) => Double
	def gradient(afterFirstArg:Boolean):kernelDerivative
	def gradientAt(afterFirstArg:Boolean,points:(DenseVector[Double],DenseVector[Double])):DenseVector[Double]
	def changeHyperParams(dv:DenseVector[Double]):KernelFunc
	def hyperParams:KernelFuncHyperParams
  }


  case class GaussianRbfParams(signalVar:Double,lengthScales:DenseVector[Double],noiseVar:Double) extends KernelFuncHyperParams{
	def getAtPosition(i: Int): Double = {
	  i match {
		case 1 => signalVar
		case i_ if i_ < lengthScales.length + 2 => lengthScales(i_ - 2)
		case i_ if i_ == lengthScales.length + 2 => noiseVar
	  }
	}

	override def toDenseVector: DenseVector[Double] = {
	  DenseVector.tabulate(lengthScales.length+2){
		index => getAtPosition(index+1)
	  }
	}

	override def fromDenseVector(dv: DenseVector[Double]): GaussianRbfParams = {
	  require(dv.length == lengthScales.length + 2)
	  val (newSignalVar,newLs,newNoiseVar) = (dv(0),dv(1 to -2),dv(-1))
	  this.copy(signalVar = newSignalVar,lengthScales = newLs,noiseVar = newNoiseVar)
	}
	
  }
  //function of form k(x_p,x_q) = signalVar^2*exp(-0.5*t(x_p-x_q)*diag(lengthScales^-2)*(x_p-x_q)) + (noiseVar^2)*(p == q)
  case class GaussianRbfKernel(rbfParams:GaussianRbfParams) extends KernelFunc {

	private val (signalVar,lengthScales,noiseVar) = (rbfParams.signalVar,rbfParams.lengthScales,rbfParams.noiseVar)

  	def apply(obj1:featureVector,obj2:featureVector,sameIndex:Boolean):Double = {
	  val valWithoutNoise:Double = signalVar*signalVar*exp(-0.5*inputWithLsProduct(obj1,obj2))
	  val retValue = if (!sameIndex){valWithoutNoise} else {
		valWithoutNoise + noiseVar*noiseVar
	  }
	  retValue
	}

	def hyperParametersNum: Int = rbfParams.lengthScales.length + 2

	def derAfterHyperParam(paramNum: Int):
		(KernelRequisites.featureVector, KernelRequisites.featureVector,Boolean) => Double = {
	  		case (vec1,vec2,sameIndex) =>
	    paramNum match {
		  case 1 => 2*signalVar*exp(-0.5*inputWithLsProduct(vec1,vec2))
		  case i_ if i_ < lengthScales.length + 2 =>
			val diffAtPosi:Double = vec1(i_ - 2)-vec2(i_ - 2)
		  	math.pow(signalVar,2)*exp(-0.5*inputWithLsProduct(vec1,vec2))* math.pow(diffAtPosi, 2) *math.pow(lengthScales(i_ - 2),-3)
		  case i_ if i_ == lengthScales.length + 2 => if (sameIndex){2*noiseVar} else {0.}
		}
	}

	def changeHyperParams(dv: DenseVector[Double]): GaussianRbfKernel = {
	  val newHyperParams = rbfParams.fromDenseVector(dv)
	  GaussianRbfKernel(newHyperParams)
	}

	def hyperParams: KernelFuncHyperParams = rbfParams

	def gradientAt(afterFirstArg: Boolean, points: (DenseVector[Double], DenseVector[Double])): DenseVector[Double] = {
	  gradient(afterFirstArg)(points._1,points._2)
	}

	def gradient(afterFirstArg: Boolean):
		(KernelRequisites.featureVector, KernelRequisites.featureVector) => DenseVector[Double] = {
	  {case (vec1,vec2) =>
		val diff = (vec1 - vec2)
		val inversedSqLs:DenseVector[Double] = lengthScales.map(ls => 1./(ls*ls))
		val a1:Double = apply(vec1,vec2,sameIndex = false)
		if (afterFirstArg){(diff :* inversedSqLs) :* (-a1)} else {(diff :* inversedSqLs) :* a1}
	  }
	}

	private def inputWithLsProduct(obj1:featureVector,obj2:featureVector):Double = {
	  val diff = (obj1 - obj2)
	  val inversedSqLs:DenseVector[Double] = lengthScales.map(ls => 1./(ls*ls))
	  (diff :* inversedSqLs) dot diff
	}
  }

  def testTrainKernelMatrix[T](test:DenseMatrix[Double],train:DenseMatrix[Double],kernelFun:KernelFunc):kernelMatrixType = {
	val result:kernelMatrixType = DenseMatrix.zeros[Double](test.rows,train.rows)
	for (i <- 0.until(test.rows)){
	  for (j <- 0.until(train.rows)){
		result(i until (i+1),j until (j+1)) := kernelFun(test(i,::).t,train(j,::).t,false)
	  }
	}
	result
  }

}
