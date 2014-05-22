package dynamicalsystems.filtering

import gp.optimization.GPOptimizer
import gp.regression.GpPredictor
import breeze.linalg.{diag, DenseVector, DenseMatrix}
import org.slf4j.LoggerFactory
import utils.KernelRequisites.{KernelFunc, KernelFuncHyperParams}
import utils.StatsUtils._
import scala.Some
import dynamicalsystems.filtering.SsmTypeDefinitions.SeriesGenerationData

/**
 * Created by mjamroz on 29/04/14.
 */
class GPUnscentedKalmanFilter(gpOptimizer: GPOptimizer, gpPredictor: GpPredictor)
  extends UnscentedKalmanFilter(gpOptimizer) {

  import UnscentedKalmanFilter._
  import KalmanFilter._
  import GpPredictor._

  val logger = LoggerFactory.getLogger(classOf[GPUnscentedKalmanFilter])
  val kernelFunc = gpPredictor.kernelFunc
  type aLCWithHyperParams = Array[(afterLearningComponents,Option[KernelFuncHyperParams])]

  def inferHiddenState(input: UnscentedFilteringInput, params: Option[UnscentedTransformParams],
								computeLL: Boolean, optimizeGpLearning:Boolean): FilteringOutput = {

	val tMax = input.observations.cols
	val initDistr = GaussianDistribution(mean = input.initMean,sigma = input.initCov)
	val (hiddenSamples,_) = input.ssmModel.generateSeries(tMax,SeriesGenerationData(initHiddenState = Right(initDistr)))
	val (gpSsmModel,qNoiseFunc,rNoiseFunc) = learnNewSsmModelWithNoises(input.observations,hiddenSamples,optimizeGpLearning)
	inferHiddenState(input.copy(ssmModel = gpSsmModel,qNoise = qNoiseFunc,rNoise = rNoiseFunc),params,computeLL)
  }

  //TODO - try to integrate it more with corresponding function from UnscentedKalmanFilter class
  def inferWithUkfOptimWithWrtToNll(input:UnscentedFilteringInput,
									initParams:Option[UnscentedTransformParams],
									optimizeGpLearning:Boolean,rangeForParam:Range=ukfParamRange) = {

	val tMax = input.observations.cols
	val initDistr = GaussianDistribution(mean = input.initMean,sigma = input.initCov)
	val (hiddenSamples,_) = input.ssmModel.generateSeries(tMax,SeriesGenerationData(initHiddenState = Right(initDistr)))
	val (gpSsmModel,qNoiseFunc,rNoiseFunc) = learnNewSsmModelWithNoises(input.observations,hiddenSamples,optimizeGpLearning)
	val ukfInput:UnscentedFilteringInput = input.copy(ssmModel = gpSsmModel,qNoise = qNoiseFunc,rNoise = rNoiseFunc)

	val objFunction:optimization.Optimization.objectiveFunction = {
	  point:Array[Double] =>
		val unscentedParams = UnscentedTransformParams.fromVector(point)
		val out = inferHiddenState(ukfInput,Some(unscentedParams),true)
		val nll = nllOfHiddenData(trueHiddenStates = hiddenSamples,
		  hiddenMeans = out.hiddenMeans,hiddenCovs = out.hiddenCovs)
		/*We want to minimize negative log likelihood */
		if (nll == Double.NegativeInfinity){veryLowValue}
		else if (nll == Double.PositiveInfinity){-veryLowValue}
		else {nll}
	}
	val gpoInput = getGpoInput(rangeForParam)
	val (optimizedParams,_) = gpOptimizer.minimize(objFunction,gpoInput)
	inferHiddenState(input,Some(UnscentedTransformParams.fromVector(optimizedParams)),true)
  }

  private def learnNewSsmModelWithNoises(observations:DenseMatrix[Double],trueHiddenStates:DenseMatrix[Double],
								optimizeGpLearning:Boolean):
  		(SsmModel,UnscentedKalmanFilter.noiseComputationFunc,UnscentedKalmanFilter.noiseComputationFunc) = {
	val trainingDataForPredictor:DenseMatrix[Double] = trueHiddenStates.t
	val trainingDataWithoutLastObj:DenseMatrix[Double] = trainingDataForPredictor(0 to -2,::)
	val (systemFuncComponents,obsFuncComponents) = (learnSystemFunction(trueHiddenStates,optimizeGpLearning),
	  learnObsFunction(trueHiddenStates,observations,optimizeGpLearning))

	logger.info("Learning of two state functions is done")

	val gpSsmModel:SsmModel = new SsmModel {
	  override val transitionFuncImpl: SsmTypeDefinitions.transitionFunc = {
		(_,prevHiddenState,_) =>

		  val diffVector = DenseVector( systemFuncComponents.map{ case (lc,hp) =>
			gpPredictor.computePosterior(trainingDataWithoutLastObj,
			  prevHiddenState.toDenseMatrix,lc._1,lc._2,kernelFunc = getKernelFunc(hp))._1.mean(0)
		  } )
		  prevHiddenState + diffVector
	  }
	  override val observationFuncImpl: SsmTypeDefinitions.observationFunc = {
		(hiddenState,_) =>
		  DenseVector( obsFuncComponents.map { case (lc,hp) =>
			gpPredictor.computePosterior(trainingDataForPredictor,
			  hiddenState.toDenseMatrix,lc._1,lc._2,kernelFunc = getKernelFunc(hp))._1.mean(0)
		  } )
	  }
	  override val obsNoise: DenseMatrix[Double] = null
	  override val latentNoise: DenseMatrix[Double] = null
	}

	val qNoiseFunc:noiseComputationFunc = {context =>
	  computeNoiseMatrix(systemFuncComponents,trainingDataWithoutLastObj,
		context.hiddenMeans(::,context.iteration-1).toDenseMatrix)
	}
	val rNoiseFunc:noiseComputationFunc = {context =>
	  computeNoiseMatrix(obsFuncComponents,trainingDataForPredictor,
		context.firstTransformFromIteration.distribution.mean.toDenseMatrix)
	}
	(gpSsmModel,qNoiseFunc,rNoiseFunc)
  }

  /*Learning gaussian process for each hidden state dimension*/
  private def learnSystemFunction(trueHiddenStates: DenseMatrix[Double],optimizeGPL:Boolean):aLCWithHyperParams = {

	val transitionDiffs = DenseMatrix.zeros[Double](trueHiddenStates.rows,trueHiddenStates.cols-1)
	(0 until (trueHiddenStates.cols-1)).foreach {
	  stateNum => transitionDiffs(::,stateNum) := (trueHiddenStates(::,stateNum+1) - trueHiddenStates(::,stateNum))
	}
	val transposedHiddenStatesWithoutLast:DenseMatrix[Double] = (trueHiddenStates(::,0 to -2)).t
	assert(transposedHiddenStatesWithoutLast.rows == transitionDiffs.cols)
	learnInputOutput(input = transposedHiddenStatesWithoutLast,output = transitionDiffs,optimizeGPL)
  }

  private def learnObsFunction(trueHiddenStates: DenseMatrix[Double],
							   observations:DenseMatrix[Double],optimizeGPL:Boolean):aLCWithHyperParams = {

	learnInputOutput(trueHiddenStates.t,observations,optimizeGPL)
  }

  private def learnInputOutput(input:DenseMatrix[Double],output:DenseMatrix[Double],optimizeGPL:Boolean):
  	aLCWithHyperParams = {

	(0 until output.rows).foldLeft(new aLCWithHyperParams(output.rows)){
	  case (componentArray,dim) =>
	  	val lc = if (optimizeGPL){
		  val tuple = gpPredictor.preComputeComponentsWithHpOptimization(input,None,output(dim,::).toDenseVector)
		  (tuple._1,Some(tuple._2))
		} else { (gpPredictor.preComputeComponents(trainingData = input,sigmaNoise = None,
		  targets = output(dim,::).toDenseVector), None)
		}
	 	componentArray(dim) = lc; componentArray
	}
  }
  
  private def computeNoiseMatrix(learningResult:aLCWithHyperParams,
								 trainingData:DenseMatrix[Double],testData:DenseMatrix[Double]):DenseMatrix[Double] = {
	  val diagVector:DenseVector[Double] = DenseVector( learningResult.map { case (lc,hp) =>
		val sigma = gpPredictor.computePosterior(trainingData,testData,lc._1,lc._2,
		  kernelFunc = getKernelFunc(hp))._1.sigma
		assert(sigma.rows == testData.rows); sigma(0,0)
	  }
	)
	diag(diagVector)
  }

  private def getKernelFunc(hp:Option[KernelFuncHyperParams]):KernelFunc = {
	hp.map(hp => kernelFunc.changeHyperParams(hp.toDenseVector)).getOrElse(kernelFunc)
  }

}
