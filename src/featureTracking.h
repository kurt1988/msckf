#include <opencv2/opencv.hpp>
#include <string>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace cv;

#ifndef FEATURE_TRACKING_H_
#define FEATURE_TRACKING_H_


struct ImageArray
{
    string img_name;
    int frameID;
    Mat gray_image;
    ImageArray(const string & imgpath, const int frmID);
    bool read();
    bool reset(const string & imgpath, const int frmID);
};


struct XYZ_Parametrization
{
    Eigen::Vector3f position;
    Eigen::Vector3f fej_position;
    float tri_error = -1;
    void update(Eigen::Vector3f & p, Eigen::Vector3f & fej_p, float err, bool change_fej = false);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


struct PointTrack
{
    vector<int> frameIDs;
    vector<Point2f> points;
    int pointID;
    XYZ_Parametrization xyz;
    void addPointInfo(int frmID, Point2f & pts, int pID = -1);
};
  

struct InversePointTrack
{
    int frameID;
    vector<int> pointIDs;
    InversePointTrack(int frmID, const vector<int> & pIDs);
};


struct ImageTrackInfo
{
    vector<Point2f> points;
    int frameID;
    vector<int> pointIDs;
    Mat descriptors;
    Mat gray_image;
    void update(const ImageArray & im_array, const vector<Point2f> & pts, const vector<int> & pIDs, const Mat & descs = Mat());
};


class Tracker
{
    private:
        ImageTrackInfo img_track_info;
        vector<PointTrack> point_track_list;
        vector<InversePointTrack> inverse_point_track_list;
        
    public:
        Tracker();
        void initialize(ImageArray & im_array);
        void initializeWithORB(ImageArray & im_array);
        ~Tracker();
        vector<PointTrack> * getPointTrackList() {return & point_track_list;};
        vector<InversePointTrack> * getInversePointTrackList() {return & inverse_point_track_list;};
        ImageTrackInfo * getImgTrackInfo() {return & img_track_info;};
        void addPointTrack(const PointTrack & ptk);
        void addInversePointTrack(const InversePointTrack & iptk);
        void deletePointTrack(int del_pID);
        void deleteInversePointTrack(int del_frmID);
        void trackPointByOpticalFlow(ImageArray & im_array);   
        void trackPointByORB(ImageArray & im_array);   
};



    


#endif





