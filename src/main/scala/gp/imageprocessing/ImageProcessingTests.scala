package gp.imageprocessing

import org.springframework.context.support.GenericXmlApplicationContext
import breeze.linalg.{DenseVector, DenseMatrix}
import gp.classification.GpClassifier
import utils.StatsUtils
import java.io._
import gp.imageprocessing.ImageProcessingUtils.PreScalingOutput
import gp.classification.GpClassifier.AfterEstimationClassifierInput
import org.slf4j.LoggerFactory

/**
 * Created by mjamroz on 04/08/14.
 */
object ImageProcessingTests {

  val imageRootDir = "../101_ObjectCategories"
  val genericAppContext = new GenericXmlApplicationContext()
  genericAppContext.load("classpath:config/spring-context.xml")
  genericAppContext.refresh()
  val gpClassifier = genericAppContext.getBean(classOf[GpClassifier])

  val fileGenerationFunc: (String,String) => String = {case (dir1,dir2) =>
	s"${dir1}_${dir2}.dat"
  }

  val logger = LoggerFactory.getLogger(this.getClass)

  case class LoadDataSetSpec(kernelMatrix:DenseMatrix[Double],
							 testTrainKernelMatrix:DenseMatrix[Double],testKernelMatrix:DenseMatrix[Double],
							 trainTargets:DenseVector[Int],testTargets:DenseVector[Int])

  case class ClassificationResult(probabsOfClass1:DenseVector[Double],misclassified:Int,
								  avgDeviationFromRightLabel:Double,errorate:Double,
								  predictedLabels:DenseVector[Int],
								  avgRightPrediction:Double,trueLabels:DenseVector[Int]){

	override def toString = {
	  val tempBuffer = (0 until probabsOfClass1.length).foldLeft(new StringBuffer){
		case (buff,index) =>
		  val indicator = if (trueLabels(index) != predictedLabels(index)){"WRONG"} else {"OK"}
		  val line = s"$index: probab of class 1: ${probabsOfClass1(index)}, true label: ${trueLabels(index)} - $indicator"
		  buff.append(line)
		  buff.append('\n')
	  }
	  val summarization = s"[misclassified = $misclassified, errorate = $errorate, " +
		s"avg-wrong-deviation = $avgDeviationFromRightLabel, avg-right-prediction = $avgRightPrediction]"
	  tempBuffer.append(summarization).toString
	}

  }

  case class ClassificationTestSuiteResult(misclassified:(Double,Double),errorate:(Double,Double),
										   avgDeviationFromRightLabel:(Double,Double),avgRightPrediction:(Double,Double)){
	override def toString = {
	  s"[misclassified = ${misclassified._1} std ${misclassified._2}, errorate = ${errorate._1} std ${errorate._2}, " +
		s"avg-wrong-deviation = ${avgDeviationFromRightLabel._1} std ${avgDeviationFromRightLabel._2}, " +
		s"avg-right-prediction = ${avgRightPrediction._1} std ${avgRightPrediction._2}]"
	}
  }



  def repeatTestFewTimes(iterNum:Int,testTrainRatio:Double,dir1:String,dir2:String):
  (Seq[ClassificationResult],ClassificationTestSuiteResult) = {
  	val (wholeKernelMatrix,imageIndexes,targets) = if (new File(fileGenerationFunc(dir1,dir2)).exists()){
	  logger.info(s"DataLoadSpec file found, loading from file ${fileGenerationFunc(dir1,dir2)}")
	  loadDataSpec(fileGenerationFunc(dir1,dir2))
	} else {
	  logger.info(s"DataLoadSpec file not found, computing in progress...")
	  val dataLoadSpec = computeKernelMatrixForWholeDataSet(dir1,dir2)
	  saveLoadDataSpec(dataLoadSpec._1,dataLoadSpec._2,dataLoadSpec._3,fileGenerationFunc(dir1,dir2))
	  logger.info(s"DataLoadSpec computed and saved to file ${fileGenerationFunc(dir1,dir2)}")
	  dataLoadSpec
	}
	val classResults = (1 to iterNum).foldLeft(Seq.empty[ClassificationResult]){
	  case (collectedResults,num) =>
		val testInstance = prepareOneTestInstance(wholeKernelMatrix,imageIndexes,targets,testTrainRatio)
	  	val classResult = executeOneTest(testInstance)
	  	collectedResults :+ classResult
	}
	println(classResults)
	val misclassifiedVector = DenseVector(classResults.map(_.misclassified.toDouble).toArray).toDenseMatrix.t
	val errorateVector = DenseVector(classResults.map(_.errorate).toArray).toDenseMatrix.t
	val avgDeviationFromRightLabel = DenseVector(classResults.map(_.avgDeviationFromRightLabel).toArray).toDenseMatrix.t
	val avgRightPrediction = DenseVector(classResults.map(_.avgRightPrediction).toArray).toDenseMatrix.t
	val misclassified = StatsUtils.meanAndVarOfData(misclassifiedVector)
	val errorRate = StatsUtils.meanAndVarOfData(errorateVector)
	val avgDeviation = StatsUtils.meanAndVarOfData(avgDeviationFromRightLabel)
	val avgRight = StatsUtils.meanAndVarOfData(avgRightPrediction)
	val suiteResult = ClassificationTestSuiteResult(misclassified = (misclassified._1(0),math.sqrt(misclassified._2(0,0))),
	  errorate = (errorRate._1(0),math.sqrt(errorRate._2(0,0))),
	  avgDeviationFromRightLabel = (avgDeviation._1(0),math.sqrt(avgDeviation._2(0,0))),
	  avgRightPrediction = (avgRight._1(0),math.sqrt(avgRight._2(0,0))))
	(classResults,suiteResult)
  }

  def computeKernelMatrixForWholeDataSet(dir1:String,dir2:String):(DenseMatrix[Double],IndexedSeq[Int],IndexedSeq[Int]) = {
	val gramMatrixBuilder = new PMKGramMatrixBuilder
	val (dir1Path,dir2Path) = (s"$imageRootDir/$dir1",s"$imageRootDir/$dir2")
	val dir1Images = ImageProcessingUtils.loadAndConvertImagesFromDir(dir1Path)
	val dir2Images = ImageProcessingUtils.loadAndConvertImagesFromDir(dir2Path)
	val diameter = computeDiameter(dir1Images,dir2Images)
	val concatenatedImagesWithIds =
	  (dir1Images ++ dir2Images).map(_._1) zip (0 until (dir1Images.size + dir2Images.size))
	val histIndex = gramMatrixBuilder.buildHistIndex(concatenatedImagesWithIds,diameter)
	val targets = (0 until dir1Images.size).map{_ => 1} ++ (0 until dir2Images.size).map{_ => -1}
	(gramMatrixBuilder.buildGramMatrix(concatenatedImagesWithIds,histIndex,diameter),
	  concatenatedImagesWithIds.map(_._2),targets)
  }

  def prepareOneTestInstance(kernelMatrix:DenseMatrix[Double],imageIndexes:IndexedSeq[Int],targets:IndexedSeq[Int],
							 testTrainRatio:Double):LoadDataSetSpec = {
	val wholeDataSetSize = imageIndexes.length
	assert(kernelMatrix.rows == wholeDataSetSize && kernelMatrix.cols == wholeDataSetSize)
	val trainSize = (testTrainRatio * wholeDataSetSize).toInt
	val shuffledIds = util.Random.shuffle(imageIndexes)
	assert(targets.length == wholeDataSetSize)
	val wholeDsShuffledKernelMatrix = DenseMatrix.tabulate[Double](wholeDataSetSize,wholeDataSetSize){
	  case (row,col) =>
	  	val (originalRow,originalCol) = (shuffledIds(row),shuffledIds(col))
	  	kernelMatrix(originalRow,originalCol)
	}
	val shuffledTargets = DenseVector.tabulate[Int](wholeDataSetSize){i => targets(shuffledIds(i))}
	val trainKernelMatrix = wholeDsShuffledKernelMatrix(0 until trainSize,0 until trainSize)
	val testKernelMatrix = wholeDsShuffledKernelMatrix(trainSize until wholeDataSetSize,trainSize until wholeDataSetSize)
	val testTrainKernelMatrix = wholeDsShuffledKernelMatrix(trainSize until wholeDataSetSize,0 until trainSize)
	val (trainTargets,testTargets) = (shuffledTargets(0 until trainSize),shuffledTargets(trainSize until wholeDataSetSize))
	LoadDataSetSpec(kernelMatrix = trainKernelMatrix,testTrainKernelMatrix = testTrainKernelMatrix,
	  testKernelMatrix = testKernelMatrix,trainTargets = trainTargets,testTargets = testTargets)

  }

  def executeOneTest(dataSpec:LoadDataSetSpec):ClassificationResult = {
	val afterEstimationParams = AfterEstimationClassifierInput(learnParams = None,
	  targets = dataSpec.trainTargets,hyperParams = null,trainKernelMatrix = dataSpec.kernelMatrix,
	  testTrainKernelMatrix = dataSpec.testTrainKernelMatrix,testKernelMatrix = dataSpec.testKernelMatrix)
	val probabsOfClassEq1 = gpClassifier.classify(afterEstimationParams)
	assert(probabsOfClassEq1.length == dataSpec.testTargets.length)
	val predictedLabels = DenseVector.zeros[Int](probabsOfClassEq1.length)
	val (misclassified,wrongDeviationSum,rightSum) =
	  (0 until probabsOfClassEq1.length).foldLeft((0,0.,0.)) { case ((missClassNum,wrongDeviation,avgRight),index) =>
		val probabOfClass1 = probabsOfClassEq1(index)
		val labelForIndex = if (probabOfClass1 >= 0.5){1} else {-1}
		predictedLabels.update(index,labelForIndex)
		val missClassified = if (labelForIndex != dataSpec.testTargets(index)){true} else {false}
		val newMissClassNum = if (missClassified){missClassNum+1}else{missClassNum}
		val newWrongDev = if (missClassified){wrongDeviation + (math.abs(probabOfClass1-0.5))} else {wrongDeviation}
		val newAvgRight = if (!missClassified){
		  val probabToAdd = if (labelForIndex == 1){probabOfClass1} else {1 - probabOfClass1}
		  avgRight + probabToAdd
		}
		else {avgRight}
		(newMissClassNum,newWrongDev,newAvgRight)
	  }
	ClassificationResult(probabsOfClass1 = probabsOfClassEq1,misclassified = misclassified,
	  avgDeviationFromRightLabel = wrongDeviationSum/misclassified,predictedLabels = predictedLabels,
	  avgRightPrediction = rightSum / (predictedLabels.length - misclassified),
	  trueLabels = dataSpec.testTargets,errorate = misclassified.toDouble / predictedLabels.length)
  }



  def computeDiameter(firstImageSet:IndexedSeq[(Array[Double],PreScalingOutput)],
						secondImageSet:IndexedSeq[(Array[Double],PreScalingOutput)]):Double = {
	(firstImageSet ++ secondImageSet).map {
	  case (_,scalingOut) => scalingOut.maxValue * scalingOut.scalingFactor
	} max
  }

  def saveLoadDataSpec(kernelMatrix:DenseMatrix[Double],ids:IndexedSeq[Int],targets:IndexedSeq[Int],fileName:String) = {
	try{
	  val fileOutStream = new FileOutputStream(new File(fileName))
	  val objOutStream = new ObjectOutputStream(fileOutStream)
	  val tupleToBeSaved = (kernelMatrix,ids,targets)
	  objOutStream.writeObject(tupleToBeSaved)
	} catch {
	  case e:Exception => throw e
	}
  }

  def loadDataSpec(fileName:String):(DenseMatrix[Double],IndexedSeq[Int],IndexedSeq[Int]) = {
	try{
	  val fileInStream = new FileInputStream(new File(fileName))
	  val objInStream = new ObjectInputStream(fileInStream)
	  objInStream.readObject().asInstanceOf[(DenseMatrix[Double],IndexedSeq[Int],IndexedSeq[Int])]
	} catch {
	  case e:Exception => throw e
	}
  }

  def main(args:Array[String]):Unit = {
	//val (_,testSuiteResult) = repeatTestFewTimes(10,0.8,"dolphin","emu")
	//val (_,testSuiteResult) = repeatTestFewTimes(10,0.9,"panda","okapi")
	val (_,testSuiteResult) = repeatTestFewTimes(100,0.9,"scissors","starfish")
	println(testSuiteResult)
  }

}
