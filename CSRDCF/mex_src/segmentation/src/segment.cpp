#include "segment.h"

//-------------------- HISTOGRAM CLASS --------------------
Histogram::Histogram(int numDimensions, int numBinsPerDimension)
{
    m_numBinsPerDim = numBinsPerDimension;
    m_numDim = numDimensions;
    p_size = std::pow(m_numBinsPerDim, m_numDim);
    p_bins.resize(p_size, 0);
    p_dimIdCoef.resize(m_numDim, 1);
    for (int i = 0; i < m_numDim-1; ++i)
        p_dimIdCoef[i] = std::pow(numBinsPerDimension, m_numDim - 1 - i);

}

void Histogram::extractForegroundHistogram(std::vector<cv::Mat> & imgChannels, cv::Mat weights, bool useMatWeights, int x1, int y1, int x2, int y2)
{
    //just for code clarity
    cv::Mat & img = imgChannels[0];

    if (!useMatWeights){
        //weights are epanechnikov distr. with peek at the center of the image;
        double cx = x1 + (x2-x1)/2.;
        double cy = y1 + (y2-y1)/2.;
        double kernelSize_width = 1.0/(0.5*static_cast<double>(x2-x1)*1.4142+1);  //sqrt(2)
        double kernelSize_height = 1.0/(0.5*static_cast<double>(y2-y1)*1.4142+1);

        cv::Mat kernelWeight(img.rows, img.cols, CV_64FC1);
        for (int y = y1; y < y2+1; ++y){
            double * weightPtr = kernelWeight.ptr<double>(y);
            double tmp_y = std::pow((cy-y)*kernelSize_height, 2);
            for (int x = x1; x < x2+1; ++x){
                weightPtr[x] = kernelProfile_Epanechnikov(std::pow((cx-x)*kernelSize_width,2) + tmp_y);
            }
        }
        weights = kernelWeight;
    }

    //extract pixel values and compute histogram
    double rangePerBinInverse = static_cast<double>(m_numBinsPerDim)/256.0;  // 1 / ( imgRange/numBinsPerDim )
    double sum = 0;
    for (int y = y1; y < y2+1; ++y){
        std::vector<const uchar *> dataPtr(m_numDim);
        for (int dim = 0; dim < m_numDim; ++dim)
            dataPtr[dim] = imgChannels[dim].ptr<uchar>(y);
        const double * weightPtr = weights.ptr<double>(y);

        for (int x = x1; x < x2+1; ++x){
            int id = 0;
            for (int dim = 0; dim < m_numDim; ++dim){
                id += p_dimIdCoef[dim]*std::floor(rangePerBinInverse*dataPtr[dim][x]);
            }
            p_bins[id] += weightPtr[x];
            sum += weightPtr[x];
        }
    }

    //normalize
    sum = 1./sum;
    for(int i = 0; i < p_size; ++i)
        p_bins[i] *= sum;
}

void Histogram::extractBackGroundHistogram(std::vector<cv::Mat> & imgChannels, int x1, int y1, int x2, int y2, int outer_x1, int outer_y1, int outer_x2, int outer_y2)
{

    //extract pixel values and compute histogram
    double rangePerBinInverse = static_cast<double>(m_numBinsPerDim)/256.0;  // 1 / ( imgRange/numBinsPerDim )
    double sum = 0;
    for (int y = outer_y1; y < outer_y2; ++y){

        std::vector<const uchar *> dataPtr(m_numDim);
        for (int dim = 0; dim < m_numDim; ++dim)
            dataPtr[dim] = imgChannels[dim].ptr<uchar>(y);

        for (int x = outer_x1; x < outer_x2; ++x){
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2)
                continue;

            int id = 0;
            for (int dim = 0; dim < m_numDim; ++dim){
                id += p_dimIdCoef[dim]*std::floor(rangePerBinInverse*dataPtr[dim][x]);
            }
            p_bins[id] += 1.0;
            sum += 1.0;
        }
    }

    //normalize
    sum = 1./sum;
    for(int i = 0; i < p_size; ++i)
        p_bins[i] *= sum;
}

cv::Mat Histogram::backProject(std::vector<cv::Mat> & imgChannels)
{
    //just for code clarity
    cv::Mat & img = imgChannels[0];

    cv::Mat backProject(img.rows, img.cols, CV_64FC1);
    double rangePerBinInverse = static_cast<double>(m_numBinsPerDim)/256.0;  // 1 / ( imgRange/numBinsPerDim )

    for (int y = 0; y < img.rows; ++y){
        double * backProjectPtr = backProject.ptr<double>(y);
        std::vector<const uchar *> dataPtr(m_numDim);
        for (int dim = 0; dim < m_numDim; ++dim)
            dataPtr[dim] = imgChannels[dim].ptr<uchar>(y);

        for (int x = 0; x < img.cols; ++x){
            int id = 0;
            for (int dim = 0; dim < m_numDim; ++dim){
                id += p_dimIdCoef[dim]*std::floor(rangePerBinInverse*dataPtr[dim][x]);
            }
            backProjectPtr[x] = p_bins[id];
        }
    }

    return backProject;
}

//-------------------- SEGMENT CLASS --------------------
std::pair<cv::Mat, cv::Mat> Segment::computePosteriors(std::vector<cv::Mat> &imgChannels, cv::Mat fgPrior, cv::Mat bgPrior, Histogram hist_target, Histogram hist_background, int numBinsPerChannel)
{
    //preprocess and normalize all data
    assert(imgChannels.size() > 0);

    //fit target to the image
    int x1 = 0;
    int y1 = 0;
    int x2 = imgChannels[0].cols-1;
    int y2 = imgChannels[0].rows-1;

    //compute resize factor so that we control the max area ~32^2
    double factor = sqrt(1000./((x2-x1)*(y2-y1)));
    //double factor = 1;
    if (factor > 1)
        factor = 1.0;
    cv::Size newSize((x2-x1)*factor, (y2-y1)*factor);

    //rescale input data
    cv::Rect roiRect_inner = cv::Rect(x1, y1, x2-x1+1, y2-y1+1);
    std::vector<cv::Mat> imgChannelsROI_inner(imgChannels.size());
    for (size_t i = 0; i < imgChannels.size(); ++i)
        cv::resize(imgChannels[i](roiRect_inner), imgChannelsROI_inner[i], newSize);

    //initialize priors if there is no external source and rescale
    cv::Mat fgPriorScaled;
    if (fgPrior.cols == 0)
        fgPriorScaled = 0.5*cv::Mat::ones(newSize, CV_64FC1);
    else
        cv::resize(fgPrior(roiRect_inner), fgPriorScaled, newSize);

    cv::Mat bgPriorScaled;
    if (bgPrior.cols == 0)
        bgPriorScaled = 0.5*cv::Mat::ones(newSize, CV_64FC1);
    else
        cv::resize(bgPrior(roiRect_inner), bgPriorScaled, newSize);

    //backproject pixels likelihood
    cv::Mat foregroundLikelihood = hist_target.backProject(imgChannelsROI_inner).mul(fgPriorScaled);
    cv::Mat backgroundLikelihood = hist_background.backProject(imgChannelsROI_inner).mul(bgPriorScaled);

    //prior for posterior, relative to the number of pixels in bg and fg
    double p_b = 5./3.;
	double p_o = 1./(p_b + 1);
    
    //convert likelihoods to posterior prob. (Bayes rule)
    cv::Mat prob_o(newSize, foregroundLikelihood.type());
    prob_o = p_o*foregroundLikelihood / (p_o*foregroundLikelihood + p_b*backgroundLikelihood);
    cv::Mat prob_b = 1.0 - prob_o;

    std::pair<cv::Mat, cv::Mat> sizedProbs = getRegularizedSegmentation(prob_o, prob_b, fgPriorScaled, bgPriorScaled);

    //resize probs to original size
    std::pair<cv::Mat, cv::Mat> probs;
    cv::resize(sizedProbs.first, probs.first, cv::Size(roiRect_inner.width, roiRect_inner.height));
    cv::resize(sizedProbs.second, probs.second, cv::Size(roiRect_inner.width, roiRect_inner.height));

    return probs;
}

std::pair<cv::Mat, cv::Mat> Segment::getRegularizedSegmentation(cv::Mat &prob_o, cv::Mat &prob_b, cv::Mat & prior_o, cv::Mat & prior_b)
{
    int hsize = std::max(1., std::floor(static_cast<double>(prob_b.cols)*3./50. + 0.5));
    int lambdaSize = hsize*2+1;

    //compute gaussian kernel
    cv::Mat lambda(lambdaSize, lambdaSize, CV_64FC1);
    double std2 = std::pow(hsize/3.0, 2);
    double sumLambda = 0.0;
    for (int y = -hsize; y < hsize + 1; ++y){
        double * lambdaPtr = lambda.ptr<double>(y+hsize);
        double tmp_y = y*y;
        for (int x = -hsize; x < hsize +1; ++x){
            double tmp_gauss = gaussian(x*x, tmp_y, std2);
            lambdaPtr[x+hsize] = tmp_gauss;
            sumLambda += tmp_gauss;
        }
    }
    sumLambda -= lambda.at<double>(hsize, hsize);
    //set center of kernel to 0
    lambda.at<double>(hsize, hsize) = 0.0;
    sumLambda = 1.0/sumLambda;
    //normalize kernel to sum to 1
    lambda = lambda*sumLambda;

    //create lambda2 kernel
    cv::Mat lambda2 = lambda.clone();
    lambda2.at<double>(hsize, hsize) = 1.0;

    double terminateThr = 1e-1;
    double logLike = std::numeric_limits<double>::max();
    int maxIter = 50;

    //return values
    cv::Mat Qsum_o(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat Qsum_b(prior_o.rows, prior_o.cols, prior_o.type());

    //algorithm temporal
    cv::Mat Si_o(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat Si_b(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat Ssum_o(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat Ssum_b(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat Qi_o(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat Qi_b(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat logQo(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat logQb(prior_o.rows, prior_o.cols, prior_o.type());

    int i;
    for (i = 0; i < maxIter; ++i){
        //follows the equations from Kristan et al. ACCV2014 paper
        //"A graphical model for rapid obstacle image-map estimation from unmanned surface vehicles"
        cv::Mat P_Io = prior_o.mul(prob_o) + std::numeric_limits<double>::epsilon();
        cv::Mat P_Ib = prior_b.mul(prob_b) + std::numeric_limits<double>::epsilon();

        cv::filter2D(prior_o, Si_o, -1, lambda, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(prior_b, Si_b, -1, lambda, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        Si_o = Si_o.mul(prior_o);
        Si_b = Si_b.mul(prior_b);
        cv::Mat normSi = 1.0/(Si_o + Si_b);
        Si_o = Si_o.mul(normSi);
        Si_b = Si_b.mul(normSi);
        cv::filter2D(Si_o, Ssum_o, -1, lambda2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(Si_b, Ssum_b, -1, lambda2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        cv::filter2D(P_Io, Qi_o, -1, lambda, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(P_Ib, Qi_b, -1, lambda, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        Qi_o = Qi_o.mul(P_Io);
        Qi_b = Qi_b.mul(P_Ib);
        cv::Mat normQi = 1.0/(Qi_o + Qi_b);
        Qi_o = Qi_o.mul(normQi);
        Qi_b = Qi_b.mul(normQi);
        cv::filter2D(Qi_o, Qsum_o, -1, lambda2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(Qi_b, Qsum_b, -1, lambda2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        prior_o = (Qsum_o + Ssum_o)*0.25;
        prior_b = (Qsum_b + Ssum_b)*0.25;
        cv::Mat normPI = 1.0/(prior_o + prior_b);
        prior_o = prior_o.mul(normPI);
        prior_b = prior_b.mul(normPI);

        //converge ?
        cv::log(Qsum_o, logQo);
        cv::log(Qsum_b, logQb);
        cv::Scalar mean = cv::sum(logQo+logQb);
        double logLikeNew = -mean.val[0]/(2*Qsum_o.rows*Qsum_o.cols);
        if (std::abs(logLike - logLikeNew) < terminateThr)
            break;
        logLike = logLikeNew;
    }
//    std::cout << "GraphCuts converges in " << i+1 << " iterations." << std::endl;
    return std::pair<cv::Mat, cv::Mat>(Qsum_o, Qsum_b);
}

// add new methods
std::vector<double> Histogram::getHistogramVector() {
    return p_bins;
}

void Histogram::setHistogramVector(double *vector) {
    for (int i=0; i<p_bins.size(); i++) {
        p_bins[i] = vector[i];
    }
}
