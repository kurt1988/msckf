#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

#ifndef COMMON_DEF_H_
#define COMMON_DEF_H_

namespace sensor_cfg
{
    const static float fx = 458.654;
    const static float cx = 367.215;
    const static float fy = 457.296;
    const static float cy = 248.375;
    const static float ZERO = 0.0;
    const static float ONE = 1.0;

    const static cv::Matx33f CAM_INTRINSIC_MAT{fx, ZERO, cx, ZERO, fy, cy, ZERO, ZERO, ONE};
    const static cv::Matx33f INV_CAM_INTRINSIC_MAT(CAM_INTRINSIC_MAT.inv());
    const static cv::Vec<float, 4> CAM_DISTORT_COEF{-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
    const static array<float, 4> IMU_NOISE{2.0000e-3, 1.6968e-04, 3.0000e-3, 1.9393e-05}; // na, ng, nba, nbg

    const static float GRAVITY{9.81};
    const static float IMG_NOISE{1.0};

    const static cv::Matx33f ROTATION_MATRIX_B2C{0.0148655429817940, 0.999557249008346, -0.0257744366974403,
                                            -0.999880929698575, 0.0149672133247192, 0.00375618835796697,
                                            0.00414029679422404, 0.0257155299479660, 0.999660727177902};
    const static cv::Vec<float, 3> POSITION_B2C{0.0652229095355311, -0.0207063854927194, -0.00805460246002952};
};


namespace algo_cfg
{
    // filter control
    const static int FIRST_BLOCK_LEN = 15;
    const static int MAX_SLIDE_NUMBER = 10;
    const static int SLIDE_SIZE = 6;
    const static bool DO_QR = true;
    
    // add KF conditions
    const static int COVISION = 100;
    const static int PARRALAX = 7;
    const static int KF_LEAST_FRAMES = 1;

    // delete KF conditions
    const static int DIST_TO_CURRENT = 3;
    const static int TRACK_LENGTH = 3;
};


namespace imgproc_cfg
{
    const static float ORBSCALE = 1.2;
    const static int ORBOCTAVE = 4;
    
    // params for goodFeaturesToTrack
    const static bool DO_UNDISTORT = true;
    const static int MAX_CORNERS = 300;
    const static int MIN_DISTANCE = 1;
    const static float QUALITY_LEVEL = 0.01;

    //LK params
    const static Size WIN_SIZE(17, 17);
    const static int MAX_LEVEL = 3;

    // optical flow track control params
    const static bool BI_DIRECTION_CHECK = true;
    const static float BI_DIST_THRE = 0.1;
    const static bool MAX_LENGTH_CONSTRAINT = true;
    const static int MAX_TRACK_LENGTH = 20;
    const static int MIN_TRACK_NUMBER = 400;

    // xyz triangulation  
    const static int KF_NUMBER = 3;
    const static float MIN_DEPTH = 0.5;
    const static float MAX_DEPTH = 15;
    const static float REPROJECTION_ERROR = 2;
};


#endif



