package breeze.test

import breeze.linalg._
import breeze.numerics._

/**
 * Created with IntelliJ IDEA.
 * User: mjamroz
 * Date: 17/11/13
 * Time: 12:53
 * To change this template use File | Settings | File Templates.
 */
class VectorTest {

  val x = DenseVector.zeros[Double](5)
  val y = DenseVector.zeros[Double](5)
  y(1 to 4) := DenseVector(.4,.3,.5,.6)

  val z:Double = x dot (y)

  println(z)


}
