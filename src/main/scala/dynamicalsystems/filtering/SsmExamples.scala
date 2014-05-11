package dynamicalsystems.filtering

import breeze.numerics.sigmoid
import utils.NumericalUtils
import breeze.linalg.DenseVector

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
  }

  class KitagawaSsm extends SsmModel {

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
