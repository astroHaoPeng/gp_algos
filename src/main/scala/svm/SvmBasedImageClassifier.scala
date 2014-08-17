package svm

import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction
import breeze.linalg.{DenseVector, DenseMatrix}
import svm.SvmBasedImageClassifier.PMKKernelForSvm
import edu.berkeley.compbio.jlibsvm.binary.{BinaryModel, C_SVC, BinaryClassificationProblemImpl}
import scala.collection.JavaConversions._
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint

/**
 * Created by mjamroz on 11/08/14.
 */
class SvmBasedImageClassifier(wholeKernelMatrix:DenseMatrix[Double]) {

  type label = Integer
  type objectType = Int

  def classifyImages(trainingIndexes:DenseVector[Int],targets:DenseVector[Int],
					 testIndexes:DenseVector[Int]):DenseVector[Int] = {

	val pmkKernel = new PMKKernelForSvm(wholeKernelMatrix)
	val examples:Map[objectType,label] = (0 until trainingIndexes.length).foldLeft(Map.empty[objectType,label]){
	  case (map,index) =>
		map + (trainingIndexes(index) -> targets(index))
	}
	val exampleIds:Map[objectType,Integer] = (0 until trainingIndexes.length).foldLeft(Map.empty[Int,Integer]){
	  case (map,index) =>
		map + (trainingIndexes(index) -> new Integer(trainingIndexes(index)))
	}
	val binaryClassProblem = new BinaryClassificationProblemImpl[label,objectType](classOf[label],examples,exampleIds)
	val builder = new ImmutableSvmParameterPoint.Builder[label,objectType]()
	builder.kernel = pmkKernel
	builder.eps = 0.02f
	val svmParam = new ImmutableSvmParameterPoint[label,objectType](builder)
	val binaryClassifier = new C_SVC[label,objectType]()
	val binaryModel:BinaryModel[label,objectType] = binaryClassifier.train(binaryClassProblem,svmParam)
	DenseVector.tabulate[Int](testIndexes.length){case index =>
		binaryModel.predictLabel(testIndexes(index))
	}
  }

}

object SvmBasedImageClassifier {

  class PMKKernelForSvm(wholeKernelMatrix:DenseMatrix[Double]) extends KernelFunction[Int]{

	override def evaluate(x: Int, y: Int): Double = {
		require(x >= 0 && x < wholeKernelMatrix.rows)
	  	require(y >= 0 && y < wholeKernelMatrix.cols)
	  	wholeKernelMatrix(x,y)
	}
  }

}
