package dynamicalsystems.filtering

import breeze.numerics.sigmoid
import utils.NumericalUtils

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

}
