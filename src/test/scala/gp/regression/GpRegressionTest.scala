package gp.regression

import org.scalatest.{BeforeAndAfterAll, WordSpec}
import breeze.numerics.{sqrt, exp}
import utils.KernelRequisites.{GaussianRbfKernel, GaussianRbfParams}
import breeze.linalg.{DenseVector, DenseMatrix}
import utils.{NumericalUtils, IOUtilities}
import gp.regression.GpRegression.PredictionInput

/**
 * Created by mjamroz on 07/04/14.
 */
class GpRegressionTest extends WordSpec with BeforeAndAfterAll{

  import NumericalUtils._

  implicit val precision = Precision(p = 0.01)

  //val (alpha,gamma,beta) = (exp(4.1),exp(-5.),10.12)
  //val (alpha,gamma,beta) = (exp(4.1),0.1,10.12)
  val (alpha,gamma,beta) = (1.,-1.,0.)
  val defaultRbfParams:GaussianRbfParams = GaussianRbfParams(alpha = alpha,gamma = gamma,beta = beta)
  val gaussianKernel = GaussianRbfKernel(defaultRbfParams)

  var trainData:DenseMatrix[Double] = _
  var targets:DenseVector[Double] = _

  //#8 example - 27.10
  val testExample = DenseVector(0.14455,  12.50,   7.870,  0.,  0.5240,  6.1720,  96.10,  5.9505,   5.,  311.0,  15.20,
	396.90,  19.15)
  //#99 example - 43.80
  val testExample1 = DenseVector( 0.08187,   0.00,   2.890,  0.,  0.4450,  7.8200,  36.90,  3.4952,   2.,  276.0,
	18.00, 393.53,   3.57)
  val gpPredictor = new GpRegression(gaussianKernel)

  override def beforeAll = {
	val data = IOUtilities.csvFileToDenseMatrix("boston.csv",sep=' ')
	targets = data(::,data.cols-1)
	trainData = data(::,0 until data.cols-1)
  }

  "GP predictor" should {

	"predict the output of train example equal to the true value in noiseless case" in {

	  	val input = PredictionInput(trainingData = trainData,testData = testExample.toDenseMatrix,
		  sigmaNoise = None,targets = targets,initHyperParams = defaultRbfParams)
	    val (distr,logLikelihood) = gpPredictor.predict(input)
	  	//assert(logLikelihood < 0)
	  	//assert(distr.mean(0) ~= 27.10)
	  	//assert(distr.sigma(0,0) ~= 0.0)
	  	println(s"distr=$distr - ll=$logLikelihood")
	}

	"optimize hyper params and predict values at specified point" in {

	  	val input = PredictionInput(trainingData = trainData,testData = testExample.toDenseMatrix,
		  sigmaNoise = None,targets = targets,initHyperParams = defaultRbfParams)
	  	val (distr,logLikelihood) = gpPredictor.predictWithParamsOptimization(input)
	  	println(s"distr=$distr - ll=$logLikelihood")
	}

  }

}
