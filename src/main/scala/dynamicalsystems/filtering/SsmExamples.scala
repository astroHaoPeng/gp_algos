package dynamicalsystems.filtering

import breeze.numerics.sigmoid
import utils.NumericalUtils
import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Created by mjamroz on 25/04/14.
 */
object SsmExamples {

  import NumericalUtils._

  class SinusoidalSsm extends SsmModel {

	override val observationFuncImpl: SsmTypeDefinitions.observationFunc = {
	  (hiddenState,_) => sigmoid(hiddenState :/ 3.)
	}

	override val transitionFuncImpl: SsmTypeDefinitions.transitionFunc = {
	  (_,previousState,_) => sin(previousState)
	}
	override val obsNoise: DenseMatrix[Double] = DenseMatrix((0.1*0.1))
	override val latentNoise: DenseMatrix[Double] = DenseMatrix((0.1*0.1))
  }

  class KitagawaSsm extends SsmModel {

	override val latentNoise:DenseMatrix[Double] = DenseMatrix((0.01*0.01))
	override val obsNoise:DenseMatrix[Double] = DenseMatrix((0.2*0.2))

	override val observationFuncImpl: SsmTypeDefinitions.observationFunc = {
	  (hiddenState,_) =>
		sin(hiddenState :* 2.) :* 5.

	}
	override val transitionFuncImpl: SsmTypeDefinitions.transitionFunc = {
	  (_,previousState,_) =>
		val denom:DenseVector[Double] = (DenseVector.ones[Double](previousState.length) :+ (previousState :* previousState))
		(previousState :* 0.5) :+ ((previousState :* 25.) :/ denom )
	}
  }

}
