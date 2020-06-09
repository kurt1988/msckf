#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <string>
using namespace std;

#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

struct IMU
{
    vector<double> ts;
    vector< Eigen::Vector3f > acc;
    vector< Eigen::Vector3f > gyro;
    IMU getData(double ts_begin, double ts_end);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct IMGInfo
{
    vector< double > ts;
    vector< string > name;
    vector< int > frameIDs;
    string path;
    // frameID should be continous; if not, redefine the following two funcs.
    double getTimestamp(int frmID);
    string getPath(int frmID);
};

struct GroundTruth
{
    vector<double> ts;
    vector< Eigen::Matrix<float, 4, 1> > q;
    vector< Eigen::Vector3f > v;
    vector< Eigen::Vector3f > p;
    vector< Eigen::Vector3f > ba;
    vector< Eigen::Vector3f > bg;
    GroundTruth getData(double ts_current);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

};


class DataLoader
{
    private:
        string root_path;
        IMU imu;
        IMGInfo img_info;
        GroundTruth gt;
    public:
        DataLoader(const string & path);
        DataLoader();
        ~DataLoader();
        void loadAndSynch(bool reset_frame = true, bool trip_name = true);
        void loadIMU(const string & imupath);
        void loadIMGInfo(const string & imgpath);
        void loadGroundTruth(const string & gtpath);
        void synchronize();
        GroundTruth * getGT() {return & gt;};
        IMU * getIMU() {return & imu;};
        IMGInfo * getIMGInfo() {return & img_info;};
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

#endif


