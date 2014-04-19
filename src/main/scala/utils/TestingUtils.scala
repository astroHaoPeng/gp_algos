package utils

/**
 * Created by mjamroz on 19/04/14.
 */
object TestingUtils {

  class ScalaObjectsCreator{

	def none[T]:Option[T] = None

	def some[T](arg:T):Option[_] = Some(arg)

  }

}
