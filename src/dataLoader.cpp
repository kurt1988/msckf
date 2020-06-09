#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include "dataLoader.h"

using namespace std;

template <typename Type>
void erase_data_vector(vector<Type> & vec, const vector<double> & ts, double ts_begin, double ts_end);

// definitions for DataLoader 
DataLoader::DataLoader()
{
    root_path = "/media/psf/Home/Documents/AlgoData/publicDataSet/MAV/MH_01_easy";
}


DataLoader::DataLoader(const string & path)
{
    root_path = path;
}


void DataLoader::loadAndSynch(bool reset_frame, bool  trip_name)
{
    string imupath = root_path + "/imu0/data.csv";
    loadIMU(imupath);

    string imgpath = root_path + "/cam0";
    loadIMGInfo(imgpath);

    string gtpath = root_path + "/state_groundtruth_estimate0/data.csv";
    loadGroundTruth(gtpath);

    synchronize();
    if (reset_frame)
    {
        int frmID0 = img_info.frameIDs[0];
        for(auto & frmID : img_info.frameIDs)
            frmID -= frmID0;
    }   
    if (trip_name)
    {
        for (auto & str : img_info.name)
            str.pop_back();
    }
}


void DataLoader::loadIMU(const string & imupath)
{
    ifstream imu_stream(imupath);
    if (imu_stream.is_open() == false)
    {
        cerr<<"no imu file opened";
        exit(EXIT_FAILURE);
    }
    string oneline;
    getline(imu_stream, oneline);
    getline(imu_stream, oneline);
    while (imu_stream)
    {
        istringstream readstr(oneline);
        string str_ts;
        getline(readstr, str_ts, ',');
        double tmp_ts;
        tmp_ts = 1.0e-9 * atof(str_ts.c_str());
        imu.ts.push_back(tmp_ts);
        Eigen::Vector3f tmp_gyro;
        Eigen::Vector3f tmp_acc;
        for (int j0 = 0; j0 < 3; j0++)
        {
            getline(readstr, str_ts, ',');
            tmp_gyro(j0) = atof(str_ts.c_str());
        }
        for (int j0 = 0; j0 < 3; j0++)
        {
            getline(readstr, str_ts, ',');
            tmp_acc(j0) = atof(str_ts.c_str());
        }
        imu.acc.push_back(tmp_acc);
        imu.gyro.push_back(tmp_gyro);
        getline(imu_stream, oneline);
    }
    imu_stream.close();
}


void DataLoader::loadIMGInfo(const string & imgpath)
{
    string imginfopath = imgpath + "/data.csv";
    ifstream imginfo_stream(imginfopath);
    if (imginfo_stream.is_open() == false)
    {
        cerr<<"no img_info file opened";
        exit(EXIT_FAILURE);
    }
    string oneline;
    getline(imginfo_stream, oneline);
    getline(imginfo_stream, oneline);
    int frmID(0);
    while (imginfo_stream)
    {
        istringstream readstr(oneline);
        string str_ts;
        getline(readstr, str_ts, ',');
        double tmp_ts;
        tmp_ts = 1.0e-9 * atof(str_ts.c_str());
        img_info.ts.push_back(tmp_ts);
        string tmp_name;
        getline(readstr, tmp_name);
        img_info.name.push_back(tmp_name);
        img_info.frameIDs.push_back(frmID++);
        getline(imginfo_stream, oneline);
    }
    imginfo_stream.close();
    img_info.path = imgpath + "/data/";
}


void DataLoader::loadGroundTruth(const string & gtpath)
{
    ifstream gtstream(gtpath);
    if (gtstream.is_open() == false)
    {
        cerr<<"no img_info file opened";
        exit(EXIT_FAILURE);
    }
    string oneline;
    getline(gtstream, oneline);
    getline(gtstream, oneline);
    while (gtstream)
    {
        istringstream readstr(oneline);
        string str_ts;
        getline(readstr, str_ts, ',');
        double tmp_ts;
        tmp_ts = 1.0e-9 * atof(str_ts.c_str());
        gt.ts.push_back(tmp_ts);
        Eigen::Vector3f posi;
        Eigen::Vector3f vel;
        Eigen::Matrix<float, 4, 1> quat;
        Eigen::Vector3f ba;
        Eigen::Vector3f bg;
        for (int j0 = 0; j0 < 16; j0++)
        {
            getline(readstr, str_ts, ',');
            float val = atof(str_ts.c_str());
            if (j0 < 3)
                posi(j0) = val;
            else if (j0 >= 3 && j0 < 7)
                quat(j0-3) = val;
            else if (j0 >= 7 && j0 < 10)
                vel(j0-7) = val;
            else if (j0 >= 10 && j0 < 13)
                bg(j0-10) = val;
            else
                ba(j0-13) = val;
        }
        gt.p.push_back(posi);
        gt.q.push_back(quat);
        gt.v.push_back(vel);
        gt.ba.push_back(ba);
        gt.bg.push_back(bg);
        getline(gtstream, oneline);
    }
    gtstream.close();
}


void DataLoader::synchronize()
{
    double ts_begin = max({imu.ts[0], gt.ts[0], img_info.ts[0]}) + 1.0e-6;
    double ts_end = min({imu.ts.back(), gt.ts.back(), img_info.ts.back()}) - 1.0e-6;

    auto copy_imu_ts = imu.ts;
    erase_data_vector(imu.acc, copy_imu_ts, ts_begin, ts_end);
    erase_data_vector(imu.gyro, copy_imu_ts, ts_begin, ts_end);
    erase_data_vector(imu.ts, copy_imu_ts, ts_begin, ts_end);

    auto copy_gt_ts = gt.ts;
    erase_data_vector(gt.ts, copy_gt_ts, ts_begin, ts_end);
    erase_data_vector(gt.p, copy_gt_ts, ts_begin, ts_end);
    erase_data_vector(gt.q, copy_gt_ts, ts_begin, ts_end);
    erase_data_vector(gt.v, copy_gt_ts, ts_begin, ts_end);
    erase_data_vector(gt.ba, copy_gt_ts, ts_begin, ts_end);
    erase_data_vector(gt.bg, copy_gt_ts, ts_begin, ts_end);

    auto copy_img_ts = img_info.ts;
    erase_data_vector(img_info.ts, copy_img_ts, ts_begin, ts_end);
    erase_data_vector(img_info.name, copy_img_ts, ts_begin, ts_end);
    erase_data_vector(img_info.frameIDs, copy_img_ts, ts_begin, ts_end);
    /*
    cout<<fixed<<endl;
    cout<<"---"<<imu.ts.front()<<'\t'<<imu.ts.back()<<'\t'<<imu.ts.back()-imu.ts.front()<<endl;
    cout<<"---"<<img_info.ts.front()<<'\t'<<img_info.ts.back()<<'\t'<<img_info.ts.back()-img_info.ts.front()<<endl;
    cout<<"---"<<gt.ts.front()<<'\t'<<gt.ts.back()<<'\t'<<gt.ts.back()-gt.ts.front()<<endl;
    */
}


DataLoader::~DataLoader()
{
}


template <typename Type>
void erase_data_vector(vector<Type> & vec, const vector<double> & vec_ts, double ts_begin, double ts_end)
{
    auto ts_ptr = find_if(vec_ts.cbegin(), vec_ts.cend(), [ts_begin](double ts){return ts >= ts_begin;});
    int idx_begin = ts_ptr - vec_ts.cbegin();
    ts_ptr = find_if(vec_ts.cbegin(), vec_ts.cend(), [ts_end](double ts){return ts >= ts_end;});
    int idx_end = ts_ptr - vec_ts.cbegin();
    if (idx_end != vec.cend() - vec.cbegin())
    {
        vec.erase(vec.cbegin() + idx_end, vec.cend());
    }
    if (idx_begin > 0)
    {
        vec.erase(vec.cbegin(), vec.cbegin() + idx_begin);
    }
}


// definitions for IMU 
IMU IMU::getData(double ts_begin, double ts_end)
{
    if (ts_end < ts_begin - 1.0e-5)
    {
        cerr<<"ts_end < ts_begin"<<endl;
        exit(EXIT_FAILURE);
    }
    IMU sub_imu;
    auto ptr = find_if(ts.cbegin(), ts.cend(), [ts_begin](double ts){return ts > ts_begin - 1.0e-5;});
    int idx_begin = ptr - ts.cbegin();
    ptr = find_if(ts.cbegin(), ts.cend(), [ts_end](double ts){return ts > ts_end + 1.0e-5;});
    int idx_end = ptr - ts.cbegin();
    if (ptr != ts.cend())
    {
        sub_imu.ts.insert(sub_imu.ts.cend(), ts.begin()+idx_begin, ts.begin()+idx_end);
        sub_imu.acc.insert(sub_imu.acc.cend(), acc.cbegin()+idx_begin, acc.cbegin()+idx_end);
        sub_imu.gyro.insert(sub_imu.gyro.cend(), gyro.cbegin()+idx_begin, gyro.cbegin()+idx_end);
    }
    return sub_imu;
}


// definitions for IMGInfo 
double IMGInfo::getTimestamp(int frmID)
{
    if (frmID > frameIDs.back() || frmID < frameIDs.front())
    {
        cerr<<"input frmaeID > max";
        exit(EXIT_FAILURE);
    }
    string imgpath;
    return ts[frmID - frameIDs[0]];
}


string IMGInfo::getPath(int frmID)
{
    if (frmID > frameIDs.back() || frmID < frameIDs.front())
    {
        cerr<<"input frmaeID > max";
        exit(EXIT_FAILURE);
    }
    string imgpath;
    imgpath = path + name[frmID - frameIDs[0]];
    return imgpath;
}


// definitions for GroundTruth 
GroundTruth GroundTruth::getData(double ts_current)
{
    GroundTruth sub_gt;
    auto ptr = find_if(ts.cbegin(), ts.cend(), 
            [ts_current](double ts){return (ts > ts_current - 1.0e-5) && (ts < ts_current + 1.0e-5);});
    int idx_begin = ptr - ts.cbegin();
    int idx_end = idx_begin + 1;
    if (ptr != ts.cend())
    {
        sub_gt.ts.insert(sub_gt.ts.cend(), ts.cbegin()+idx_begin, ts.cbegin()+idx_end);
        sub_gt.p.insert(sub_gt.p.cend(), p.cbegin()+idx_begin, p.cbegin()+idx_end);
        sub_gt.q.insert(sub_gt.q.cend(), q.cbegin()+idx_begin, q.cbegin()+idx_end);
        sub_gt.v.insert(sub_gt.v.cend(), v.cbegin()+idx_begin, v.cbegin()+idx_end);
        sub_gt.ba.insert(sub_gt.ba.cend(), ba.cbegin()+idx_begin, ba.cbegin()+idx_end);
        sub_gt.bg.insert(sub_gt.bg.cend(), bg.cbegin()+idx_begin, bg.cbegin()+idx_end);
    }
    return sub_gt;
}




