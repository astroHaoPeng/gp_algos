package gp.classification

import gp.classification.GpClassifier.ClassifierInput
import scala.collection.immutable.NumericRange
import breeze.linalg.DenseVector
import org.slf4j.LoggerFactory
import java.io.{PrintWriter, File}

/**
 * Created by mjamroz on 27/03/14.
 */
class MeshHyperParamsLogLikelihoodEvaluator(likelihoodEvaluator:MarginalLikelihoodEvaluator) {

  val logger = LoggerFactory.getLogger(this.getClass)

  import MeshHyperParamsLogLikelihoodEvaluator._

  def evaluate(hyperParamsRanges:IndexedSeq[NumericRange[Double]],classificationContext:ClassifierInput):HyperParamsMeshValues = {

  	val initHyperParams:Array[Double] = hyperParamsRanges.map(_.start).toArray
	val result = new HyperParamsMeshValues
	recEvaluate(initHyperParams,hyperParamsRanges,0,classificationContext,result)
	result
  }
  
  private def recEvaluate(currentHyperParams:Array[Double],allRanges:IndexedSeq[NumericRange[Double]],
				  rangeIndex:Int,classContext:ClassifierInput,result:HyperParamsMeshValues):Unit = {

	for (hyperParam <- allRanges(rangeIndex)){
	  val copiedHyperParams = currentHyperParams.clone()
	  copiedHyperParams(rangeIndex) = hyperParam
	  if (rangeIndex + 1 < allRanges.length){
		recEvaluate(copiedHyperParams,allRanges,rangeIndex+1,classContext,result)
	  }
	  val likelihood:Double = likelihoodEvaluator.logLikelihoodWithoutGrad(classContext.trainData.get,classContext.targets,
		DenseVector(currentHyperParams))
	  logger.info(s"Evaluated likelihood for hyperParams = ${DenseVector(currentHyperParams)}, value = ${likelihood}")
	  result.addResult(copiedHyperParams,likelihood)
	}	
  }


}


object MeshHyperParamsLogLikelihoodEvaluator {

  import scala.collection._

  class HyperParamsMeshValues{

	val paramsLikelihood:mutable.Map[Array[Double],Double] = mutable.Map()

	def addResult(params:Array[Double],logLikelihood:Double){
		paramsLikelihood.put(params,logLikelihood)
	}

	def getLikelihood(params:Array[Double]):Option[Double] = {
		paramsLikelihood.get(params)
	}

	override def toString:String = {
	  (paramsLikelihood.foldLeft(new StringBuffer("")){
		  case (buffer,(params,likelihood)) =>
		  	buffer.append(s"${DenseVector(params)}  -> ${likelihood}\n")
		}).toString
	}

	def writeToFile(fileName:String):Unit = {
	  writeToFile(new File(fileName))
	}

	def writeToFile(file:File):Unit = {
	  	var printWriter:PrintWriter = null
	  	try{
		  printWriter = new PrintWriter(file)
		  val string = paramsLikelihood.foldLeft(new StringBuffer("")){
			case (buffer,(params,likelihood)) =>
			  params.foreach(paramValue => buffer.append(s"${paramValue}\t"))
			  buffer.append(s"${likelihood}\n")
		  }.toString
		  printWriter.write(string)
		} catch {
		  case e:Exception => println(e)
		} finally {
		  printWriter.close()
		}
	}

  }

}
