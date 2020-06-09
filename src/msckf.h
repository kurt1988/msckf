#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include "featureTracking.h"
#include "commonConfig.h"
#include "dataLoader.h"

using namespace Eigen;
using namespace std;

#ifndef MSCKF_H_
#define MSCKF_H_


struct Slide
{
    Matrix3f R;
    Vector3f p;
    int frameID;
    Vector3f fej_p;
    Vector3f fej_v;
    template <typename _Type>
    void copyFrom(const _Type & sld);
    void update(Matrix3f & new_R, Vector3f & new_p);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


struct FirstBlock
{
    Matrix3f R;
    Vector3f v;
    Vector3f p;
    Vector3f ba;
    Vector3f bg;
    Vector3f fej_p;
    Vector3f fej_v;
    int frameID;
    MatrixXf cov;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


struct ProjectionInformation
{
    VectorXf error;
    MatrixXf Jx;
};


class State
{
    private:
        FirstBlock first_block;
        vector<Slide> slides;
        int slide_number = 0;
        Tracker tracker;
    public:
        State();
        ~State();
        void initializeWithGT(DataLoader & data, int init_frmID);
        void predict(IMU & imu_data, int curr_frmID);
        void addSilde(const Slide & sld);
        void deleteSlide();
        bool decideKeyFrame();
        void deletePointTracks();
        void deleteLostPointTracks();
        void deleteInversePointTracks();
        void triangulation();
        void projection(ProjectionInformation & proj_info);
        void visualCorrection();
        FirstBlock * getFirstBlock() {return & first_block;};
        vector<Slide> * getSlides() {return & slides;};
        Tracker * getTracker() {return & tracker;};
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
};



#endif


