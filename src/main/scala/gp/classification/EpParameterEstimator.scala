package gp.classification

import breeze.linalg.{cholesky, diag, DenseVector, DenseMatrix}
import breeze.stats.distributions.Gaussian
import breeze.numerics.{abs, log, sqrt}
import org.slf4j.LoggerFactory

/**
 * Created by mjamroz on 11/03/14.
 */
class EpParameterEstimator(kernelMatrix:DenseMatrix[Double],targets:DenseVector[Int],
						   stopCriterion:EpParameterEstimator.stopCriterionFunc) {

  val logger = LoggerFactory.getLogger(classOf[EpParameterEstimator])

  import utils.MatrixUtils._
  import utils.StatsUtils._
  import EpParameterEstimator._

  require(kernelMatrix.rows == targets.length)
  val gaussianHelper = Gaussian(mu = 0,sigma = 1)


  /**
   *
   * @return tuple containing estimated parameters and lower triangular matrix from cholesky decomposition.
   *         This matrix is useful later - to avoid unnecessary computations
   */
  def estimateSiteParams:(SiteParams,DenseMatrix[Double]) = {

	val n = kernelMatrix.rows
	var (tauSiteParams,niSiteParams,miParams) = (DenseVector.zeros[Double](n),DenseVector.zeros[Double](n),
	  DenseVector.zeros[Double](n))
	val (cavDistrTauParams,cavDistrNiParams) = (tauSiteParams.copy,niSiteParams.copy)
	var sigmaMatrix = kernelMatrix.copy
	var lowerTriangular:DenseMatrix[Double] = null
	var (currentParams,oldParams) = (SiteParams(niSiteParams = niSiteParams,tauSiteParams = tauSiteParams),
	  SiteParams(niSiteParams = niSiteParams,tauSiteParams = tauSiteParams))

	for (j <- Stream.from(0).takeWhile { j => (j == 0 || !stopCriterion(EpEstimationContext(currentParams = currentParams,oldParams = oldParams)))}){
	  oldParams = SiteParams(tauSiteParams = tauSiteParams.copy,niSiteParams = niSiteParams.copy)
	  logger.info(s"Beginning of iteration = ${j}")

	  for (i <- (0 until n)) {
		cavDistrTauParams.update(i,1/sigmaMatrix(i,i) - tauSiteParams(i))
		cavDistrNiParams.update(i,miParams(i)/sigmaMatrix(i,i) - niSiteParams(i))
		val (miMarginalParam,sigmaMarginalParam,_) =
		  marginalMoments(cavDistrNiParams(i)/cavDistrTauParams(i),1/cavDistrTauParams(i),targets(i))
		val tauSiteParamDiff = 1/sigmaMarginalParam - cavDistrTauParams(i) - tauSiteParams(i)
		tauSiteParams.update(i,tauSiteParams(i) + tauSiteParamDiff)
		niSiteParams.update(i,miMarginalParam/sigmaMarginalParam - cavDistrNiParams(i))
		val ithColumnOfSigmaMatrix = sigmaMatrix(::,i)
		sigmaMatrix -= ((ithColumnOfSigmaMatrix * ithColumnOfSigmaMatrix.t) :* (1/(1/tauSiteParamDiff + sigmaMatrix(i,i))))
		miParams = sigmaMatrix * niSiteParams
	  }
	  val tauDiagVector:DenseVector[Double] = sqrt(tauSiteParams)
	  lowerTriangular =
		cholesky(DenseMatrix.eye[Double](n) + ((tauDiagVector*tauDiagVector.t) :* kernelMatrix))
	  val vMatrix = forwardSolve(L = lowerTriangular,b = cloneCols(tauDiagVector,n) :* kernelMatrix)
	  sigmaMatrix = kernelMatrix - (vMatrix.t * vMatrix)
	  miParams = sigmaMatrix * niSiteParams
	  currentParams = currentParams.copy(niSiteParams = niSiteParams,tauSiteParams = tauSiteParams)

	  logger.info(s"End of iteration = ${j} - average difference = ${avgBetweenSiteParams(oldParams,currentParams)}")
	}
	val siteParams = SiteParams(niSiteParams = niSiteParams,tauSiteParams = tauSiteParams)
	val cavityParams = CavityDistributionParams(tauParams = cavDistrTauParams,niParams = cavDistrNiParams)
	val logMarginalLikelihood:Double= epMarginalLikelihood(siteParams,cavityParams,lowerTriangular,sigmaMatrix)
	(siteParams.copy(marginalLogLikelihood = Some(logMarginalLikelihood)),lowerTriangular)
  }

  def epMarginalLikelihood(siteParams:SiteParams,cavityParams:CavityDistributionParams,
								   lowerTriangular:DenseMatrix[Double],sigmaMatrix:DenseMatrix[Double]):Double = {

	val (tauSiteParams,cavityTauParams) = (siteParams.tauSiteParams,cavityParams.tauParams)
	val n = siteParams.tauSiteParams.length
	val cavityMiParams:DenseVector[Double] = cavityParams.niParams/cavityParams.tauParams

	val tauSiteCavityParamsSum:DenseVector[Double] = tauSiteParams + cavityTauParams
	val tauSiteCavityParamsSumDiagMatrix = diag(1 / tauSiteCavityParamsSum)
	val tauSiteCavityParamsSumDiagVec = (1 / tauSiteCavityParamsSum)
	val temp2:DenseMatrix[Double] = sigmaMatrix - tauSiteCavityParamsSumDiagMatrix
	val fifthAndSecondTermFirst:DenseVector[Double] = (siteParams.niSiteParams.t * temp2) * siteParams.niSiteParams
	val temp3:DenseVector[Double] = ((cavityMiParams :* cavityTauParams) :* tauSiteCavityParamsSumDiagVec)
	val temp4:DenseVector[Double] = ((tauSiteParams :* cavityMiParams) - (siteParams.niSiteParams :* 2.))
	val fifthAndSecondTermSecond:Double = temp3.toDenseVector dot temp4

	val (thirdTerm,fourthAndFirstTerm) = (0 until n).foldLeft((0.,0.)){
	  case ((thirdTerm,fourthAndFirstTerm),i) => 
		val newThirdTerm = thirdTerm + 
			log(pnorm(targets(i)*cavityMiParams(i)/sqrt(1 + 1/cavityParams.tauParams(i))))
	  	val newFourthAndFirstTerm = fourthAndFirstTerm
		+ 0.5*log(1+siteParams.tauSiteParams(i)/cavityParams.tauParams(i)) - log(lowerTriangular(i,i))
		(newThirdTerm,newFourthAndFirstTerm)
	}
	thirdTerm + fourthAndFirstTerm + 0.5*(fifthAndSecondTermFirst(0) + fifthAndSecondTermSecond)
  }

  private def marginalMoments(cavDistrMiParam:Double,cavDistrSigmaParam:Double,target:Int)
  	:(Double,Double,Double) = {

	val temp = sqrt(1+cavDistrSigmaParam)
	val z = (target * cavDistrMiParam)/temp
	val (dnormZ,pnormZ) = (dnorm(z),pnorm(z))
	val zMarginalParam = pnormZ
	val miMarginalParam = cavDistrMiParam + (target*cavDistrSigmaParam*dnormZ)/(pnormZ*temp)
	val sigmaMarginalParam = cavDistrSigmaParam -
	  ((cavDistrSigmaParam*cavDistrSigmaParam*dnormZ)*(z+dnormZ/pnormZ))/((1+cavDistrSigmaParam)*pnormZ)
	(miMarginalParam,sigmaMarginalParam,zMarginalParam)
  }


  /** R code:
   *  marginalMoments function
   *
   *  temp <- sqrt(1+cavity_distr_sigma_param)
	z <- (y*cavity_distr_mi_param)/temp
	dnorm_z <- dnorm(z)
	pnorm_z <- pnorm(z)
	z_marignal_param <- pnorm_z
	mi_marginal_param <- cavity_distr_mi_param + (y*cavity_distr_sigma_param*dnorm_z)/(pnorm_z*temp)
	sigma_marignal_param <- cavity_distr_sigma_param - ((cavity_distr_sigma_param^2)*dnorm_z)*(z+dnorm_z/pnorm_z)/((1+cavity_distr_sigma_param)*pnorm_z)
	list(marginal_mi=mi_marginal_param,marginal_sigma=sigma_marignal_param,marginal_z=z_marignal_param)

	* epMarginalLikelihood function
	*
	* tau_site_param <- site_params$tau_param
	cavity_tau_param <- cavity_distr_params$tau_param
	cavity_mi_param <- cavity_distr_params$mi_param
	ni_site_param <- site_params$ni_param
	chol_fact <- other_useful_params$chol_fact
	sigma_param <- other_useful_params$sigma_param
	fourth_and_first_term <- 0
	third_term <- 0
	fifth_and_second_term <- 0.5*t(ni_site_param) %*% (sigma_param - diag(1/(tau_site_param + cavity_tau_param))) %*% ni_site_param
	n <- length(tau_site_param)
	for (i in 1:n){
	  third_term <- third_term + log(pnorm(targets[i]*cavity_mi_param[i]/sqrt(1+1/cavity_tau_param[i])))
	  fourth_and_first_term <- fourth_and_first_term + 0.5*log(1+tau_site_param[i]/cavity_tau_param[i]) - log(chol_fact[i,i])
	}
	third_term + fourth_and_first_term + fifth_and_second_term

	* estimateSiteParams function
	*
	* ni_site_param <- tau_site_param <- mi_param <- cavity_distr_tau_param <- cavity_distr_ni_param <- array(0,n)
	sigma_param <- kernelMatrix
	#for now - very straightforward convergence criterium
	for (j in 1:5){
	  cat(sprintf("iteration %d\n",j))
	  for (i in 1:n){
		cavity_distr_tau_param[i] <- 1/sigma_param[i,i]- tau_site_param[i]
		cavity_distr_ni_param[i] <- mi_param[i]/sigma_param[i,i] - ni_site_param[i]
		#tau_param = delta_param^(-1), mi_param = ni_param/tau_param
		marginal_m <- marginal_moments(cavity_distr_ni_param[i]/cavity_distr_tau_param[i],cavity_distr_tau_param[i]^(-1),targets[i])
		tau_site_param_diff <- marginal_m$marginal_sigma^(-1) - cavity_distr_tau_param[i] - tau_site_param[i]
		tau_site_param[i] <- tau_site_param[i] + tau_site_param_diff
		ni_site_param[i] <- marginal_m$marginal_mi/marginal_m$marginal_sigma - cavity_distr_ni_param[i]
		sigma_param <- sigma_param - (1/(1/tau_site_param_diff + sigma_param[i,i]))*(sigma_param[,i] %*% t(sigma_param[,i]))
		mi_param <- sigma_param %*% ni_site_param
	  }
	  tau_diag_vector <- as.numeric(sqrt(tau_site_param))
	  low_triang <- t(chol(diag(n) + outer(tau_diag_vector,tau_diag_vector)*kernelMatrix))
	  v_matrix <- forwardsolve(l=low_triang,x=tau_diag_vector * kernelMatrix)
	  old_sigma <- sigma_param
	  sigma_param <- kernelMatrix - (t(v_matrix) %*% v_matrix)
	  mi_param <- sigma_param %*% ni_site_param

	  marginal_likelihood <- ep_marginal_likelihood_appr(list(tau_param=tau_site_param,ni_param=ni_site_param),
	  list(tau_param=cavity_distr_tau_param,mi_param=cavity_distr_ni_param/cavity_distr_tau_param),targets,
	  list(chol_fact=low_triang,sigma_param=sigma_param))
		  list(ni_site_param=as.vector(ni_site_param),tau_site_param=tau_site_param,marginal_likelihood=marginal_likelihood)

	*/

}


object EpParameterEstimator {

  type stopCriterionFunc = EpEstimationContext => Boolean

  case class SiteParams(tauSiteParams:DenseVector[Double],niSiteParams:DenseVector[Double],
						marginalLogLikelihood:Option[Double] = None)
  case class CavityDistributionParams(tauParams:DenseVector[Double],niParams:DenseVector[Double])

  case class EpEstimationContext(oldParams:SiteParams,currentParams:SiteParams)

  class AvgBasedStopCriterion(eps:Double) extends stopCriterionFunc{

	def apply(context: EpEstimationContext): Boolean = {
	  val (oldParams,currentParams) = (context.oldParams,context.currentParams)
	  abs(avgBetweenSiteParams(oldParams,currentParams)) < eps
	}
  }

  def avgBetweenSiteParams(oldParams:SiteParams,currentParams:SiteParams):Double = {
	val sumOfDiffsOfParams = 0.until(oldParams.niSiteParams.length).foldLeft(0.){
	  case (avg,index) => avg +
		(currentParams.niSiteParams(index) - oldParams.niSiteParams(index)) +
		(currentParams.tauSiteParams(index) - oldParams.tauSiteParams(index))
	}
	(sumOfDiffsOfParams/2*currentParams.niSiteParams.length)
  }

}
