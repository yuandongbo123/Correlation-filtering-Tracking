#include <opencv2/opencv.hpp>
#include "segment.h"
#include "mex.h"
#include "mex_utils.hpp"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] ) {

	if(nrhs < 3) mexErrMsgTxt("Valid input: [image], [x1, y1, x2, y2], bins.");
    if(nlhs > 1) mexErrMsgTxt("Too many oputput arguments.");

    cv::Mat img = toMat(prhs[0], CV_8U, true);
    std::vector<cv::Mat> imgChannels;
    cv::split(img, imgChannels);

    // object region
    cv::Mat region = toMat(prhs[1], CV_32S, false);
    int x1 = std::min(imgChannels[0].cols-1, std::max(0, region.at<int>(0, 0)));
    int x2 = std::min(imgChannels[0].cols-1, std::max(0, region.at<int>(2, 0)));
    int y1 = std::min(imgChannels[0].rows-1, std::max(0, region.at<int>(1, 0)));
    int y2 = std::min(imgChannels[0].rows-1, std::max(0, region.at<int>(3, 0)));

    int bins = static_cast<int>(mxGetScalar(prhs[2]));


    int offsetX = (x2-x1+1)/2;
    int offsetY = (y2-y1+1)/2;

    int outer_y1 = std::min(imgChannels[0].rows-1, std::max(0, (int)(y1-offsetY)));
    int outer_y2 = std::min(imgChannels[0].rows-1, std::max(0, (int)(y2+offsetY+1)));
    int outer_x1 = std::min(imgChannels[0].cols-1, std::max(0, (int)(x1-offsetX)));
    int outer_x2 = std::min(imgChannels[0].cols-1, std::max(0, (int)(x2+offsetX+1)));

    Histogram hist_background(imgChannels.size(), bins);
    hist_background.extractBackGroundHistogram(imgChannels, x1, y1, x2, y2,
            outer_x1, outer_y1, outer_x2, outer_y2);

    // convert histogram p_bins to opencv vector that will be converted to
    // matlab array
    std::vector<double> histogram_vector = hist_background.getHistogramVector();
    plhs[0]= mxCreateDoubleMatrix(1, histogram_vector.size(), mxREAL);
    double *ptr_mask;
    ptr_mask = (double *) mxGetData(plhs[0]);
    for (int i=0; i<histogram_vector.size(); i++) {
        ptr_mask[i] = histogram_vector[i];
    }
}
