package utils

import breeze.linalg.{DenseVector, DenseMatrix}

/**
 * Created by mjamroz on 11/04/14.
 */
object DataSetsTestingUtils {

  case class DataLoadSpec(fileName:String,sep:Char = ' ', divRatio:Option[Double])
  
  case class DataSet(trainingData:DenseMatrix[Double],trainingLabels:DenseVector[Double],
								   testData:DenseMatrix[Double],testLabels:DenseVector[Double],
								   wholeDataSet:DenseMatrix[Double])
  
  def loadDataSet(loadSpec:DataLoadSpec):Either[Exception,DataSet] = {
	try{
	  val (divRatio,fileName) = (loadSpec.divRatio,loadSpec.fileName)
	  val ratio = divRatio.getOrElse(1.0)
	  require(ratio > 0.0 && ratio <= 1.0)
	  val wholeDataSet:DenseMatrix[Double] = IOUtilities.csvFileToDenseMatrix(fileName,sep = loadSpec.sep)
	  val dsLength:Int = wholeDataSet.rows
	  val numOfTrainCases = (ratio * wholeDataSet.rows).toInt
	  val trainData:DenseMatrix[Double] = wholeDataSet(0 until numOfTrainCases,0 until (wholeDataSet.cols-1))
	  val tempTargets:DenseVector[Double] = wholeDataSet(::,wholeDataSet.cols-1)
	  val targets:DenseVector[Double] = tempTargets(0 until numOfTrainCases)
	  val testSet:DenseMatrix[Double] = wholeDataSet(numOfTrainCases until dsLength,0 until (wholeDataSet.cols-1))
	  val testTargets:DenseVector[Double] = tempTargets(numOfTrainCases until dsLength)
	  Right(DataSet(trainingData = trainData,trainingLabels = targets,
		testData = testSet,testLabels = testTargets,
		wholeDataSet = DenseMatrix.vertcat[Double](trainData,testSet)))
	} catch {
	  case ex:Exception => Left(ex)
	}
  }

  def loadDataSet(fileName:String,sep:Char = ' '):DenseMatrix[Double] = {
	IOUtilities.csvFileToDenseMatrix(fileName,sep = sep)
  }



}
