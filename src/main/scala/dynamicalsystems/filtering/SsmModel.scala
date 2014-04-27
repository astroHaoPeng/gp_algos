package dynamicalsystems.filtering

/**
 * Created by mjamroz on 24/04/14.
 */

import SsmTypeDefinitions._
import breeze.linalg.{DenseMatrix, DenseVector}
import utils.StatsUtils.{GaussianDistribution, NormalDistributionSampler}

trait SsmModel {

  val transitionFuncImpl:transitionFunc
  val observationFuncImpl:observationFunc


  def sampleHiddenState(previousStateVal:DenseVector[Double],
						optionalInput:DenseVector[Double],qNoise:DenseMatrix[Double],iterNum:Int):DenseVector[Double] = {

	val nextStateVal = transitionFuncImpl(optionalInput,previousStateVal,iterNum)
	val noise:DenseVector[Double] = NormalDistributionSampler.sample(
	  GaussianDistribution(mean = DenseVector.zeros[Double](nextStateVal.length), sigma = qNoise))
	nextStateVal :+ noise
  }

  def sampleObservation(hiddenStateVal:DenseVector[Double],rNoise:DenseMatrix[Double],iterNum:Int):DenseVector[Double] = {

	val observationVal: DenseVector[Double] = observationFuncImpl(hiddenStateVal,iterNum)
	val noise = NormalDistributionSampler.sample(GaussianDistribution(
	  mean = DenseVector.zeros[Double](observationVal.length), sigma = rNoise))
	observationVal :+ noise
  }

  /*_1 - hidden state values, _2 - observations*/
  def generateSeries(length:Int,generationData:SeriesGenerationData):(DenseMatrix[Double],DenseMatrix[Double]) = {

	require(generationData.rNoise.length == length && generationData.qNoise.length == length,
	  "Number of noise matrices needs to be equal to series length")
	val initHiddenState:DenseVector[Double] = generationData.initHiddenState match {
	  case Left(dv) => dv
	  case Right(normalDistr) =>
		NormalDistributionSampler.sample(normalDistr)
	}
	val hiddenStateMatrix = DenseMatrix.zeros[Double](initHiddenState.length,length)
	val observationStateMatrix = DenseMatrix.zeros[Double](generationData.qNoise(0).rows,length)
	(0 until length).foldLeft(initHiddenState){
	  case (currentHiddenState,iterNum) =>
	  	val observation = sampleObservation(currentHiddenState,generationData.rNoise(iterNum),iterNum)
	  	hiddenStateMatrix(::,iterNum) := currentHiddenState
	  	observationStateMatrix(::,iterNum) := observation
	  	sampleHiddenState(currentHiddenState,null,generationData.qNoise(iterNum),iterNum)
	}

	(hiddenStateMatrix,observationStateMatrix)
  }

}
