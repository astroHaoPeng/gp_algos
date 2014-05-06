package dynamicalsystems.filtering

import gp.optimization.GPOptimizer
import gp.regression.GpPredictor
import breeze.linalg.{diag, DenseVector, DenseMatrix}
import org.slf4j.LoggerFactory

/**
 * Created by mjamroz on 29/04/14.
 */
class GPUnscentedKalmanFilter(gpOptimizer: GPOptimizer, gpPredictor: GpPredictor)
  extends UnscentedKalmanFilter(gpOptimizer) {

  import UnscentedKalmanFilter._
  import KalmanFilter._
  import GpPredictor._

  val logger = LoggerFactory.getLogger(classOf[GPUnscentedKalmanFilter])

  def inferHiddenState(input: UnscentedFilteringInput, params: Option[UnscentedTransformParams],
								trueHiddenStates: DenseMatrix[Double], computeLL: Boolean): FilteringOutput = {

	val trainingDataForPredictor:DenseMatrix[Double] = trueHiddenStates.t
	val trainingDataWithoutLastObj:DenseMatrix[Double] = trainingDataForPredictor(0 to -2,::)
	val (systemFuncComponents,obsFuncComponents) = (learnSystemFunction(trueHiddenStates),
	  learnObsFunction(trueHiddenStates,input.observations))

	logger.info("Learning of two state functions is done")

  	val gpSsmModel:SsmModel = new SsmModel {
	  override val transitionFuncImpl: SsmTypeDefinitions.transitionFunc = {
		(_,prevHiddenState,_) =>

		  val diffVector = DenseVector( systemFuncComponents.map{lc:afterLearningComponents =>
			  gpPredictor.computePosterior(trainingDataWithoutLastObj,prevHiddenState.toDenseMatrix,lc._1,lc._2)._1.mean(0)
		  } )
		  prevHiddenState + diffVector
	  }
	  override val observationFuncImpl: SsmTypeDefinitions.observationFunc = {
		(hiddenState,_) =>
		  DenseVector( obsFuncComponents.map {lc:afterLearningComponents =>
			gpPredictor.computePosterior(trainingDataForPredictor,hiddenState.toDenseMatrix,lc._1,lc._2)._1.mean(0)
		  } )
	  }

	}

	val qNoiseFunc:noiseComputationFunc = {context =>
	  computeNoiseMatrix(systemFuncComponents,trainingDataWithoutLastObj,
		context.hiddenMeans(::,context.iteration-1).toDenseMatrix)
	}
	val rNoiseFunc:noiseComputationFunc = {context =>
	  computeNoiseMatrix(obsFuncComponents,trainingDataForPredictor,
		context.firstTransformFromIteration.distribution.mean.toDenseMatrix)
	}

	inferHiddenState(input.copy(ssmModel = gpSsmModel,qNoise = qNoiseFunc,rNoise = rNoiseFunc),params,computeLL)
  }

  /*Learning gaussian process for each hidden state dimension*/
  private def learnSystemFunction(trueHiddenStates: DenseMatrix[Double]):Array[afterLearningComponents] = {

	val transitionDiffs = DenseMatrix.zeros[Double](trueHiddenStates.rows,trueHiddenStates.cols-1)
	(0 until (trueHiddenStates.cols-1)).foreach {
	  stateNum => transitionDiffs(::,stateNum) := (trueHiddenStates(::,stateNum+1) - trueHiddenStates(::,stateNum))
	}
	val transposedHiddenStatesWithoutLast:DenseMatrix[Double] = (trueHiddenStates(::,0 to -2)).t
	assert(transposedHiddenStatesWithoutLast.rows == transitionDiffs.cols)
	learnInputOutput(input = transposedHiddenStatesWithoutLast,output = transitionDiffs)
  }

  private def learnObsFunction(trueHiddenStates: DenseMatrix[Double],
							   observations:DenseMatrix[Double]):Array[afterLearningComponents] = {

	learnInputOutput(trueHiddenStates.t,observations)
  }

  private def learnInputOutput(input:DenseMatrix[Double],output:DenseMatrix[Double]):
  	Array[afterLearningComponents] = {

	(0 until output.rows).foldLeft(new Array[afterLearningComponents](output.rows)){
	  case (componentArray,dim) =>
	  	componentArray(dim) = gpPredictor.preComputeComponents(trainingData = input,sigmaNoise = None,
		  targets = output(dim,::).toDenseVector)
		componentArray
	}
  }
  
  private def computeNoiseMatrix(learningResult:Array[afterLearningComponents],
								 trainingData:DenseMatrix[Double],testData:DenseMatrix[Double]):DenseMatrix[Double] = {
	  val diagVector:DenseVector[Double] = DenseVector( learningResult.map { lc =>
		val sigma = gpPredictor.computePosterior(trainingData,testData,lc._1,lc._2)._1.sigma
		assert(sigma.rows == testData.rows); sigma(0,0)
	  }
	)
	diag(diagVector)
  }

}
