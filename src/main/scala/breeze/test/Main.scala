package breeze.test

import breeze.linalg.{DenseMatrix, DenseVector}
import gp.regression.{GpRegression}
import utils.IOUtilities
import scala.math._
import utils.KernelRequisites.{GaussianRbfKernel, GaussianRbfParams}
import gp.regression.GpRegression.PredictionInput

/**
 * Created with IntelliJ IDEA.
 * User: mjamroz
 * Date: 17/11/13
 * Time: 20:06
 * To change this template use File | Settings | File Templates.
 */
object Main {

  val (alpha,gamma,beta) = (exp(4.1),exp(-5.),10.12)
  val defaultRbfParams:GaussianRbfParams = GaussianRbfParams(alpha = alpha,gamma = gamma)
  val gaussianKernel = GaussianRbfKernel(defaultRbfParams)

  def main(args:Array[String]):Unit = {
	//#8 example
	val testExample = DenseVector(0.14455,  12.50,   7.870,  0.,  0.5240,  6.1720,  96.10,  5.9505,   5.,  311.0,  15.20,
	  396.90,  19.15)
	//#99 example
	val testExample1 = DenseVector( 0.08187,   0.00,   2.890,  0.,  0.4450,  7.8200,  36.90,  3.4952,   2.,  276.0,
	  18.00, 393.53,   3.57)
	val data =IOUtilities.csvFileToDenseMatrix("boston.csv",sep=' ')
	val targets:DenseVector[Double] = data(::,data.cols-1)
	val featureMatrix:DenseMatrix[Double] = data(::,0 until data.cols-1)
	val input = PredictionInput(trainingData = featureMatrix,testData = testExample.toDenseMatrix,
	  sigmaNoise = None,targets = targets,initHyperParams = defaultRbfParams)
	println(new GpRegression(gaussianKernel).predict(input))
  }

}
