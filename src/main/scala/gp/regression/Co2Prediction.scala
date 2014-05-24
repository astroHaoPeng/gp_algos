package gp.regression

import utils.KernelRequisites.{KernelFuncHyperParams, KernelFunc}
import utils.KernelRequisites
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{pow, sin, exp}
import org.springframework.core.io.ClassPathResource
import org.springframework.context.support.GenericXmlApplicationContext
import gp.regression.GpPredictor.PredictionInput

/**
 * Created by mjamroz on 22/05/14.
 */
object Co2Prediction {

  case class Co2HyperParams(dv: DenseVector[Double]) extends KernelFuncHyperParams {

	override def fromDenseVector(dv: DenseVector[Double]): KernelFuncHyperParams = Co2HyperParams(dv)

	override def toDenseVector: DenseVector[Double] = dv

	override def getAtPosition(i: Int): Double = dv(i - 1)

	override def toString = dv.toString()
  }

  case class Co2Kernel(co2HyperParams: Co2HyperParams) extends KernelFunc {

	import utils.KernelRequisites._

	/*k1(x1,x2) = hp1^2*exp(-(x1-x2)^2/(2*hp2^2) )
	  k2(x1,x2) = hp3^2*exp(-(x1-x2)^2/(2*hp4^2) - 2*sin(pi*(x1-x2))^2/(hp5^2)) )
	  k3(x1,x2) = hp6^2*(1 + (x1-x2)^2/(2*hp8*hp7^2))^(-hp8)
	  k4(x1,x2) = hp9^2*exp(-(x1-x2)^2/(2*hp10^2) ) + hp11^2*sameIndex
	  k(x1,x2) = k1(x1,x2) + k2(x1,x2) + k3(x1,x2) + k4(x1,x2)
	 */
	override def apply(obj1: featureVector, obj2: featureVector, sameIndex: Boolean): Double = {
	  require(obj1.length == 1 && obj2.length == 1, message = "This kernel is applicable only for 1D objects")
	  val (hp1, hp2, hp3, hp4, hp5, hp6, hp7, hp8, hp9, hp10, hp11) = getHyperParams
	  val (x1, x2) = (obj1(0), obj2(0))
	  val (xDiff, xDiffSq) = (x1 - x2, (x1 - x2) * (x1 - x2))
	  val k1Val: Double = hp1 * hp1 * exp(-xDiffSq / (2 * hp2 * hp2))
	  val sinVal: Double = sin(Math.PI * xDiff)
	  val k2Val: Double = hp3 * hp3 * exp((-xDiffSq / (2 * hp4 * hp4)) - 2 * sinVal * sinVal / (hp5 * hp5))
	  val k3Pow1: Double = 1 + xDiffSq / (2 * hp8 * hp7 * hp7)
	  val k3Val: Double = hp6 * hp6 * pow(k3Pow1, -hp8)
	  val k4Val: Double = hp9 * hp9 * exp(-xDiffSq / (2 * hp10 * hp10))
	  val indNoise: Double = if (sameIndex) {
		hp11 * hp11
	  } else {
		0.
	  }
	  k1Val + k2Val + k3Val + k4Val + indNoise
	}

	override def hyperParams: KernelFuncHyperParams = co2HyperParams

	override def changeHyperParams(dv: DenseVector[Double]): KernelFunc = Co2Kernel(Co2HyperParams(dv))

	override def gradientAt(afterFirstArg: Boolean, points: (DenseVector[Double], DenseVector[Double])): DenseVector[Double] = ???

	override def gradient(afterFirstArg: Boolean): KernelRequisites.kernelDerivative = ???

	override def derAfterHyperParam(paramNum: Int): (KernelRequisites.featureVector, KernelRequisites.featureVector, Boolean) => Double = {
	  case (vec1: featureVector, vec2: featureVector, sameIndex: Boolean) =>
		val (x1, x2) = (vec1(0), vec2(0))
		val (hp1, hp2, hp3, hp4, hp5, hp6, hp7, hp8, hp9, hp10, hp11) = getHyperParams
		paramNum match {
		  case num if (num < 3) =>
			derAfterFirstKernel(x1,x2,hp1,hp2,num)
		  case num if (num < 6) =>
			derAfterSecondKernel(x1,x2,hp3,hp4,hp5,num)
		  case num if (num < 9) =>
			derAfterThirdKernel(x1,x2,hp6,hp7,hp8,num)
		  case num if (num < 12) =>
			derAfterFourthKernel(x1,x2,hp9,hp10,hp11,num,sameIndex)

		}
	}

	override def hyperParametersNum: Int = 11

	private def getHyperParams = {
	  (co2HyperParams.getAtPosition(1), co2HyperParams.getAtPosition(2),
		co2HyperParams.getAtPosition(3), co2HyperParams.getAtPosition(4), co2HyperParams.getAtPosition(5),
		co2HyperParams.getAtPosition(6), co2HyperParams.getAtPosition(7), co2HyperParams.getAtPosition(8),
		co2HyperParams.getAtPosition(9), co2HyperParams.getAtPosition(10), co2HyperParams.getAtPosition(11))
	}

	private def derAfterFirstKernel(x1: Double, x2: Double, hp1: Double, hp2: Double, paramNum: Int): Double = {
	  val sqDiff: Double = (x1 - x2) * (x1 - x2)
	  paramNum match {
		case 1 => 2 * hp1 * exp(-sqDiff / (2 * hp2 * hp2))
		case 2 => hp1 * hp1 * exp(-sqDiff / (2 * hp2 * hp2)) * sqDiff * pow(hp2, -3)
	  }
	}

	private def derAfterSecondKernel(x1: Double, x2: Double, hp3: Double, hp4: Double, hp5: Double, paramNum: Int): Double = {
	  val (xDiff, sqDiff) = (x1 - x2, (x1 - x2) * (x1 - x2))
	  val sinVal: Double = sin(Math.PI * xDiff)
	  val k2Val: Double = hp3 * hp3 * exp(-sqDiff / (2 * hp4 * hp4) - 2 * sinVal * sinVal / (hp5 * hp5))
	  paramNum match {
		case 3 => 2*k2Val/hp3
		case 4 => k2Val*sqDiff*pow(hp4,-3)
		case 5 => k2Val*4*sinVal*sinVal*pow(hp5,-3)
	  }
	}

	private def derAfterThirdKernel(x1:Double,x2:Double,hp6:Double,hp7:Double,hp8:Double,paramNum:Int):Double = {
	  val sqDiff = (x1-x2)*(x1-x2)
	  val k3Pow1: Double = 1 + sqDiff / (2 * hp8 * hp7 * hp7)
	  paramNum match {
		case 6 => 2*hp6*pow(k3Pow1,-hp8)
		case 7 => hp6*hp6*pow(k3Pow1,-hp8-1)*sqDiff*pow(hp7,-3)
		case 8 =>
		  val firstTerm:Double = exp(-hp8*math.log(k3Pow1))
		  val secondTerm:Double = -math.log(k3Pow1) + (hp8*sqDiff/(2*hp7*hp7*hp8*hp8*k3Pow1))
		  hp6*hp6*firstTerm * secondTerm
	  }
	}

	private def derAfterFourthKernel(x1:Double,x2:Double,hp9:Double,hp10:Double,hp11:Double,paramNum:Int,sameIndex:Boolean):Double = {
	  val sqDiff = (x1-x2)*(x1-x2)
	  val k4Val: Double = hp9 * hp9 * exp(-sqDiff / (2 * hp10 * hp10))
	  paramNum match {
		case 9 => 2*k4Val/hp9
		case 10 => k4Val*sqDiff*pow(hp10,-3)
		case 11 => if (sameIndex) {2*hp11} else {0.}
	  }
	}
  }

  def loadInput:DenseMatrix[Double] = {
	val lines = io.Source.fromFile(new ClassPathResource("co2/maunaLoa.txt").getFile).getLines().toSeq
	val finalMatrix = DenseMatrix.zeros[Double](lines.length,14)
	lines.foldLeft(0){
	  case (index,line) =>
		val numbers = line.split("(\\s|\\t)").map(_.toDouble)
	  	finalMatrix(index,::) := DenseVector(numbers)
	  	index+1
	}
	finalMatrix
  }

  /*Result is Nx2 matrix where 1st column of row is a year, 2nd column of row is a co2 ppm */
  def co2DataToYearWithValue(matrix:DenseMatrix[Double]):(DenseMatrix[Double]) = {
  	val yearCo2PpmTuples:IndexedSeq[(Double,Double)] = (0 until matrix.rows).foldLeft(IndexedSeq[(Double,Double)]()){
	  case (acc,row) =>
		val year = matrix(row,0)
		val tuplesFromYear = (1 until matrix.cols-1).foldLeft(IndexedSeq[(Double,Double)]()){
			case (acc,month) =>
		  		val co2Ppm:Double = matrix(row,month)
				if (co2Ppm > 0){
				  acc :+ (year + (1/12.)*(month-1),co2Ppm)
				} else {acc}
	  }
	  acc ++ tuplesFromYear
	}
	DenseMatrix.tabulate[Double](yearCo2PpmTuples.length,2){
	  case (row,col) => if (col == 0){yearCo2PpmTuples(row)._1} else {yearCo2PpmTuples(row)._2}
	}
  }

  def main(args:Array[String]) = {
	val co2Concentration = co2DataToYearWithValue(loadInput)
	val genericContext = new GenericXmlApplicationContext()
	genericContext.load("classpath:config/spring-context.xml")
	genericContext.refresh()
	val co2GpPredictor = genericContext.getBean("co2GpPredictor",classOf[GpPredictor])
	val rbfGpPredictor = genericContext.getBean("gpPredictor",classOf[GpPredictor])
	val gpInput = PredictionInput(trainingData = co2Concentration(::,0).toDenseMatrix.t,sigmaNoise = None,
	  targets = co2Concentration(::,1),testData = DenseMatrix((2022.)))
	val predictResult = co2GpPredictor.predict(gpInput)
	val rbfPredictResult = rbfGpPredictor.predict(gpInput)
	val optimizedPredictResult = co2GpPredictor.predictWithParamsOptimization(gpInput,true)
	println(s"Co2Kernel - Posterior distribution = ${predictResult._1}, Marginal LL = ${predictResult._2}")
	println(s"RbfKernel - Posterior distribution = ${rbfPredictResult._1}, Marginal LL = ${rbfPredictResult._2}")
	println(s"Co2Kernel - HPO - Posterior distribution = ${optimizedPredictResult._1}," +
	  s" Marginal LL = ${optimizedPredictResult._2}, optimal hyperParams = ${optimizedPredictResult._3}")
  }
}
