package gp.imageprocessing

import breeze.linalg.DenseMatrix
import gp.imageprocessing.PMKUtils.Histogram
import org.slf4j.LoggerFactory

/**
 * Created by mjamroz on 04/08/14.
 */
class PMKGramMatrixBuilder {

  import collection._

  type histogramIndex = mutable.Map[Int,IndexedSeq[Histogram]]

  val logger = LoggerFactory.getLogger(classOf[PMKGramMatrixBuilder])

  def buildHistIndex(imagesWithIds:Iterable[(Array[Double],Int)],diameter:Double):histogramIndex = {

	imagesWithIds.foldLeft(mutable.Map.empty[Int,IndexedSeq[Histogram]]) {case (cache,(image,index)) =>
	  val histograms = PMKUtils.featureExtractingFunc(image,diameter)
	  cache.put(index,histograms); cache
	}
  }

  def buildGramMatrix(imagesWithIds:IndexedSeq[(Array[Double],Int)],histIndex:histogramIndex,diameter:Double):DenseMatrix[Double] = {
	
	val kernel = new PMKKernel(diameter)
	val resultGramMatrix = DenseMatrix.zeros[Double](imagesWithIds.size,imagesWithIds.size)

	for (row <- 0 until imagesWithIds.size; col <- 0 to row){
	  val (id1,id2) = (imagesWithIds(row)._2,imagesWithIds(col)._2)
	  val (histsForRow,histsForCol) = (histIndex(id1),histIndex(id2))
	  val kernelVal = if (row == col){kernel.apply(null,null,sameIndex = true)} else {
		val unormalizedVal = kernel.unormalizedKernelVal(histsForRow,histsForCol)
		kernel.normalize(histsForRow,histsForCol,unormalizedVal)
	  }
	  logger.info(s"Computed PMK value for [row = $row, col = $col] and [row = $col, col = $row]")
	  resultGramMatrix.update(row,col,kernelVal)
	  resultGramMatrix.update(col,row,kernelVal)
	}
	resultGramMatrix
  }

  def buildGramMatrix(imagesWithIds1:IndexedSeq[(Array[Double],Int)],
					  imagesWithIds2:IndexedSeq[(Array[Double],Int)],histIndex:histogramIndex,diameter:Double):DenseMatrix[Double] = {
	val kernel = new PMKKernel(diameter)
	val resultGramMatrix = DenseMatrix.zeros[Double](imagesWithIds1.size,imagesWithIds2.size)

	for (row <- 0 until imagesWithIds1.size; col <- 0 until imagesWithIds2.size){
	  val (id1,id2) = (imagesWithIds1(row)._2,imagesWithIds2(col)._2)
	  val (histsForRow,histsForCol) = (histIndex(id1),histIndex(id2))
	  val unormalizedVal = kernel.unormalizedKernelVal(histsForRow,histsForCol)
	  val kernelVal = kernel.normalize(histsForRow,histsForCol,unormalizedVal)
	  resultGramMatrix.update(row,col,kernelVal)
	  logger.info(s"Computed PMK value for [row = $row, col = $col]")
	}
	resultGramMatrix
  }

}
