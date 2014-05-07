package utils

import breeze.linalg.{DenseVector, DenseMatrix}
import org.springframework.core.io.ClassPathResource
import breeze.io.CSVReader
import java.io.{File, PrintWriter}

/**
 * Created by mjamroz on 14/03/14.
 */
object IOUtilities {

  def csvFileToDenseMatrix(file:String,sep:Char=','):DenseMatrix[Double] = {
	val input = io.Source.fromFile(new ClassPathResource(file).getFile).getLines().foldLeft(new StringBuilder()){
	  case (buff,line) => buff.append(line); buff.append('\n')
	}.toString()
	val seqFromReader = CSVReader.parse(input,separator = sep)
	val (rowSize,colSize) =  (seqFromReader.length,seqFromReader(1).filterNot(_.isEmpty).length)
	val vectorSeq = seqFromReader.foldLeft(Seq[DenseVector[Double]]()){case (collectedRows,lineFromFile) =>
	  try{
		collectedRows :+ DenseVector(lineFromFile.filterNot(_.isEmpty).map(_.toDouble).toArray)
	  } catch {
		case _:Exception => collectedRows
	  }
	}
	val result = DenseMatrix.zeros[Double](vectorSeq.length,colSize)
	vectorSeq.foldLeft(0){case (indx,vector) => result(indx,::) := vector; indx+1}
	result
  }

  def writeVectorsToFile(vec1:DenseVector[Double],vec2:DenseVector[Double],fileName:String) = {
	require(vec1.length == vec2.length,"Two vectors should have the same size")
	var printWriter:PrintWriter = null
	try{
	  printWriter = new PrintWriter(new File(fileName))
	  val strToWrite = (0 until vec1.length).foldLeft(new StringBuffer("")){
		case (buffer,index) =>
			buffer.append(s"${vec1(index)}\t${vec2(index)}\n")
	  }.toString
	  printWriter.write(strToWrite)
	} catch {
	  case e:Exception => println(e)
	} finally {
	  printWriter.close()
	}
  }
}
