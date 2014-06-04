package svm

import gp.regression.GpPredictor.PredictionInput
import utils.KernelRequisites.{KernelFunc, KernelFuncHyperParams}
import edu.berkeley.compbio.jlibsvm.regression.{EpsilonSVR, MutableRegressionProblemImpl}
import breeze.linalg.DenseVector
import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint

/**
 * Created by mjamroz on 28/05/14.
 */

//TODO - make it invariant to kernelFunction
class SvmRegressionImpl(kernelFunc:KernelFunc) {

  type objectType = DenseVector[Double]
  /*Svm returns just a value, not probabilistic distribution*/
  def predict(input:PredictionInput,
							   hyperParams:KernelFuncHyperParams=kernelFunc.hyperParams):DenseVector[Double] = {

	val trainingDataSize = input.targets.length
	val regressionProblemImpl = new MutableRegressionProblemImpl[objectType](trainingDataSize)
	populateWithData(regressionProblemImpl,input)
	val epsSvm = new EpsilonSVR[objectType,MutableRegressionProblemImpl[objectType]]
	val builder = new ImmutableSvmParameterPoint.Builder[java.lang.Float,objectType]()
	builder.kernel = kernelFunc
	builder.eps = 0.02f
	val immutableSvmParam = new ImmutableSvmParameterPoint[java.lang.Float,objectType](builder)
	val regressionModel = epsSvm.train(regressionProblemImpl,immutableSvmParam)
	DenseVector.tabulate[Double](input.testData.rows){
	  case index =>
	  	regressionModel.predictValue(input.testData(index,::).t).toDouble
	}
  }

  private def populateWithData(mutableRegProblem:MutableRegressionProblemImpl[objectType],input:PredictionInput):Unit = {
	val (training,targets) =  (input.trainingData,input.targets)
	for (i <- 0 until training.rows){
	  mutableRegProblem.addExample(training(i,::).t,targets(i).toFloat)
	}
  }

  class JLibSvmKernel(kernelFunc:KernelFunc) extends KernelFunction[objectType] {

	override def evaluate(x: objectType, y: objectType): Double = {
		val sameIndex = x.equals(y)
	    kernelFunc.apply(x,y,sameIndex)
	}
  }

  implicit def gpToSvmCo2Kernel(gpKernel:KernelFunc):KernelFunction[objectType] = {
	new JLibSvmKernel(kernelFunc)
  }

}
