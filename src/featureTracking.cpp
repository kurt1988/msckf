#include <iostream>
#include <string>
#include<algorithm>
#include "featureTracking.h"
#include "commonConfig.h"

using namespace std;
using namespace cv;

template <typename Type, typename Type1, typename Type2>
void erase_by(vector<Type> & vec, vector<uchar> & by, vector<Type1> & vec1, vector<Type2> & vec2);


// ImageArray
ImageArray::ImageArray(const string & imgpath, const int frmID)
{
    frameID = frmID;
    img_name = imgpath; 
    read();
}


bool ImageArray::read()
{
    bool read_flag;
    gray_image = imread(img_name, IMREAD_GRAYSCALE);
    if (!gray_image.data)
    {
        read_flag = false;
    }
    
    if (imgproc_cfg::DO_UNDISTORT)
    {
        Mat udst_img = Mat::zeros(gray_image.size(), CV_8UC1);
        undistort(gray_image, udst_img, sensor_cfg::CAM_INTRINSIC_MAT, sensor_cfg::CAM_DISTORT_COEF);
        udst_img.copyTo(gray_image);
    }
    read_flag = true;
    return read_flag;
}


bool ImageArray::reset(const string & imgpath, const int frmID)
{
    frameID = frmID;
    img_name = imgpath; 
    bool read_flag = read();
    return read_flag;
}


// XYZ_Parametrization
void XYZ_Parametrization::update(Eigen::Vector3f & p, Eigen::Vector3f & fej_p, float err, bool change_fej)
{
    position = p;
    tri_error = err;
    if (change_fej)
        fej_position = fej_p;
}


// PointTrack
void PointTrack::addPointInfo(int frmID, Point2f & pts, int pID)
{
    frameIDs.push_back(frmID);
    points.push_back(pts);
    if (pID >= 0)
        pointID = pID;
}


// InversePointTrack
InversePointTrack::InversePointTrack(int frmID, const vector<int> & pIDs)
{
    frameID = frmID;
    pointIDs = pIDs;
}


// ImageTrackInfo
void ImageTrackInfo::update(const ImageArray & im_array, const vector<Point2f> & pts, const vector<int> & pIDs, const Mat & descs)
{
    points = pts;
    frameID = im_array.frameID;
    gray_image = im_array.gray_image;
    pointIDs = pIDs;
    descriptors = descs;
}


// Tracker             
Tracker::Tracker()
{
}


void Tracker::initialize(ImageArray & im_array)
{
    int frmID = im_array.frameID;
    vector<Point2f> pts;
    goodFeaturesToTrack(im_array.gray_image, pts, imgproc_cfg::MAX_CORNERS, imgproc_cfg::QUALITY_LEVEL, imgproc_cfg::MIN_DISTANCE);
    vector<int> pIDs;
    for (int k0 = 0;  k0 < pts.size(); k0++)
        pIDs.push_back(k0);
    img_track_info.update(im_array, pts, pIDs);

    int k0 = 0;
    for (auto pts0: pts)
    {
        PointTrack ptk;
        ptk.addPointInfo(frmID, pts0, pIDs.at(k0++));
        addPointTrack(ptk);
    }
    InversePointTrack iptk(frmID, pIDs);
    addInversePointTrack(iptk);
}


void Tracker::initializeWithORB(ImageArray & im_array)
{
    int frmID = im_array.frameID;
    auto orb = ORB::create(imgproc_cfg::MAX_CORNERS, imgproc_cfg::ORBSCALE, imgproc_cfg::ORBOCTAVE);
    vector<KeyPoint> keypoints;
    Mat descs;
    orb->detectAndCompute(im_array.gray_image, Mat(), keypoints, descs);
    vector<int> pIDs;
    vector<Point2f> pts;
    for (int k0 = 0;  k0 < keypoints.size(); k0++)
    {
        pIDs.push_back(k0);
        pts.push_back(keypoints[k0].pt);
    }
    
    img_track_info.update(im_array, pts, pIDs, descs);

    int k0 = 0;
    for (auto pts0 : pts)
    {
        PointTrack ptk;
        ptk.addPointInfo(frmID, pts0, pIDs.at(k0++));
        addPointTrack(ptk);
    }
    InversePointTrack iptk(frmID, pIDs);
    addInversePointTrack(iptk);
}


void Tracker::addPointTrack(const PointTrack & ptk)
{
    point_track_list.push_back(ptk);
}


void Tracker::addInversePointTrack(const InversePointTrack & iptk)
{
    inverse_point_track_list.push_back(iptk);
}


void Tracker::deletePointTrack(int del_pID)
{
    auto ptr = find_if(point_track_list.cbegin(), point_track_list.cend(), 
        [del_pID](PointTrack ptk){return ptk.pointID == del_pID;});
    point_track_list.erase(ptr);
}


void Tracker::deleteInversePointTrack(int del_frmID)
{
    auto ptr = find_if(inverse_point_track_list.cbegin(), inverse_point_track_list.cend(), 
        [del_frmID](InversePointTrack iptk){return iptk.frameID == del_frmID;});
    inverse_point_track_list.erase(ptr);
}


void Tracker::trackPointByOpticalFlow(ImageArray & im_array)
{
    vector<int> store_curr_pointIDs;
    vector<Point2f> store_curr_points;

    if (imgproc_cfg::MAX_LENGTH_CONSTRAINT)
    {
        for (auto ipid = img_track_info.pointIDs.cbegin(); ipid != img_track_info.pointIDs.cend(); ipid++)
        {
            auto ptr = find_if(point_track_list.cbegin(), point_track_list.cend(),
                [ipid](auto ptk){return ptk.pointID == *ipid;});
            if (ptr->frameIDs.size() > imgproc_cfg::MAX_TRACK_LENGTH)
            {
                int lag = ipid - img_track_info.pointIDs.cbegin();
                img_track_info.pointIDs.erase(ipid);
                img_track_info.points.erase(img_track_info.points.cbegin() + lag);
                ipid--;
            }
        }
    }

    if (img_track_info.points.size() > 0)
    {
        vector<Point2f> curr_points;
        vector<uchar> status;
        vector<float> err;

        calcOpticalFlowPyrLK(img_track_info.gray_image, im_array.gray_image, img_track_info.points, curr_points, status, err,
            imgproc_cfg::WIN_SIZE, imgproc_cfg::MAX_LEVEL);

        erase_by(img_track_info.points, status, curr_points, img_track_info.pointIDs);

        if (imgproc_cfg::BI_DIRECTION_CHECK)
        {
            vector<Point2f> back_corners;
            vector<uchar> back_status;
            vector<float> back_err;
            calcOpticalFlowPyrLK(im_array.gray_image, img_track_info.gray_image, curr_points, back_corners, back_status, back_err,
                imgproc_cfg::WIN_SIZE, imgproc_cfg::MAX_LEVEL);
            
            for (auto bptr = back_corners.cbegin(); bptr != back_corners.cend(); bptr++)
            {
                bool erase_flag = false;
                int lag = bptr - back_corners.cbegin();
                if (*(back_status.cbegin() + lag) == 0)
                    erase_flag = true;
                float n2 = norm(*bptr - *(img_track_info.points.cbegin() + lag));
                if (n2 > imgproc_cfg::BI_DIST_THRE)
                    erase_flag = true;
                if (erase_flag)
                {
                    back_corners.erase(bptr);
                    back_status.erase(back_status.cbegin() + lag);
                    img_track_info.points.erase(img_track_info.points.cbegin() + lag);
                    img_track_info.pointIDs.erase(img_track_info.pointIDs.cbegin() + lag);
                    curr_points.erase(curr_points.cbegin() + lag);
                    bptr--;
                }
            }
        }

        if (img_track_info.pointIDs.size() > 0)
        {
            auto prev_points = img_track_info.points;
            auto bptr = point_track_list.begin();
            for (auto itr_pid = img_track_info.pointIDs.begin(); itr_pid != img_track_info.pointIDs.end(); itr_pid++)
            {
                auto ptr = find_if(bptr, point_track_list.end(), [itr_pid](auto ptk){return ptk.pointID == *itr_pid;});
                Point2f point = *(curr_points.begin() + (itr_pid - img_track_info.pointIDs.begin()));
                ptr->addPointInfo(im_array.frameID, point);
                bptr = ptr;
            } 

        } 

        store_curr_pointIDs.insert(store_curr_pointIDs.cend(), img_track_info.pointIDs.cbegin(), img_track_info.pointIDs.cend());
        store_curr_points.insert(store_curr_points.cend(), curr_points.cbegin(), curr_points.cend());  
    }
  
    if (store_curr_points.size() < imgproc_cfg::MIN_TRACK_NUMBER)
    {
        Mat mask = Mat(im_array.gray_image.size(), CV_8UC1, Scalar(255));
        int u, v, c0, c1, r0, r1;
        int row_num = im_array.gray_image.rows;
        int col_num = im_array.gray_image.cols;
        for (auto point : store_curr_points)
        {
            u = cvRound(point.x);
            v = cvRound(point.y);  
            c0 = max(0, u - imgproc_cfg::MIN_DISTANCE);
            c1 = min(col_num, u + imgproc_cfg::MIN_DISTANCE);
            r0 = max(0, v - imgproc_cfg::MIN_DISTANCE);
            r1 = min(row_num, v + imgproc_cfg::MIN_DISTANCE);
            mask(Range(r0, r1), Range(c0, c1)) = 0;    
        }

        vector<Point2f> added_corners;
        goodFeaturesToTrack(im_array.gray_image, added_corners, imgproc_cfg::MAX_CORNERS, imgproc_cfg::QUALITY_LEVEL, imgproc_cfg::MIN_DISTANCE, mask);
        
        int pid = point_track_list.back().pointID;
        for (auto pts : added_corners)
        {
            pid += 1;
            PointTrack ptk;
            ptk.addPointInfo(im_array.frameID, pts, pid);
            addPointTrack(ptk);
            store_curr_pointIDs.push_back(pid);
            store_curr_points.push_back(pts);
        }
    }

    InversePointTrack iptk(im_array.frameID, store_curr_pointIDs);
    addInversePointTrack(iptk);

    img_track_info.update(im_array, store_curr_points, store_curr_pointIDs);
}


void Tracker::trackPointByORB(ImageArray & im_array)
{
    vector<int> store_curr_pointIDs;
    vector<Point2f> store_curr_points;
    Mat store_curr_descs;
    auto orb = ORB::create(imgproc_cfg::MAX_CORNERS, imgproc_cfg::ORBSCALE, imgproc_cfg::ORBOCTAVE);
    Mat curr_descs;
    vector<KeyPoint> curr_key_points;
    orb->detectAndCompute(im_array.gray_image, Mat(), curr_key_points, curr_descs);

    vector<Point2f> curr_points;
    vector<int> add_idx;
    for (int k0 = 0; k0 < curr_key_points.size(); k0++)
    {
        curr_points.push_back(curr_key_points[k0].pt);
        add_idx.push_back(k0);
    }
    
    BFMatcher matcher(NORM_HAMMING, true);
    vector<DMatch> matches;
    matcher.match(curr_descs, img_track_info.descriptors, matches);
    float min_dist = 30;
    for (auto mtch : matches)
    {
        if (mtch.distance < min_dist)
            min_dist = mtch.distance;
    }
    
    vector<int> curr_match_idx, prev_match_idx;

    for (int j0 = 0; j0 < matches.size(); j0++)
    {
        if (matches[j0].distance <= max(2 * min_dist, 30.0f))
        {
            curr_match_idx.push_back(matches[j0].queryIdx);
            prev_match_idx.push_back(matches[j0].trainIdx);
            int del_idx = matches[j0].queryIdx;
            auto idx_ptr = find_if(add_idx.begin(), add_idx.end(), [del_idx](auto added_idx){return added_idx == del_idx;});
            add_idx.erase(idx_ptr);
        }
        
    }
    
    if (prev_match_idx.size() > 0)
    {
        auto prev_points = img_track_info.points;
        for (int k0 = 0; k0 < prev_match_idx.size(); k0++)
        {
            int prev_idx = prev_match_idx[k0];
            int curr_idx = curr_match_idx[k0];
            int pid = img_track_info.pointIDs[prev_idx];
            auto ptk_ptr = find_if(point_track_list.begin(), point_track_list.end(), [pid](auto ptk){return ptk.pointID == pid;});
            Point2f pt = curr_points[curr_idx];
            ptk_ptr->addPointInfo(im_array.frameID, pt);
            store_curr_pointIDs.push_back(pid);
            store_curr_points.push_back(pt);
            store_curr_descs.push_back(curr_descs.row(curr_idx));
        } 
    } 

    if (add_idx.size() > 0)
    {
        int pid = point_track_list.back().pointID;
        for (auto idx : add_idx)
        {
            pid += 1;
            PointTrack ptk;
            Point2f pt = curr_points[idx];
            ptk.addPointInfo(im_array.frameID, pt, pid);
            addPointTrack(ptk);
            store_curr_pointIDs.push_back(pid);
            store_curr_points.push_back(pt);
            store_curr_descs.push_back(curr_descs.row(idx));
        }
    }

    InversePointTrack iptk(im_array.frameID, store_curr_pointIDs);
    addInversePointTrack(iptk);

    img_track_info.update(im_array, store_curr_points, store_curr_pointIDs, store_curr_descs);
}


template <typename Type, typename Type1, typename Type2>
void erase_by(vector<Type> & vec, vector<uchar> & by, vector<Type1> & vec1, vector<Type2> & vec2)
{
    for (auto ptr = vec.cbegin(); ptr != vec.cend(); ptr++)
    {
        auto idx = ptr - vec.cbegin();
        if (*(by.cbegin() + idx) == 0)
        {
            vec.erase(ptr);
            by.erase(by.cbegin() + idx);
            vec1.erase(vec1.cbegin() + idx);
            vec2.erase(vec2.cbegin() + idx);
            ptr--;
        }
    }
}

    
Tracker::~Tracker()
{
}











