#include<Eigen/SVD>
#include<opencv2/core/eigen.hpp>
#include <algorithm>
#include <random> 
#include <chrono> 
#include "msckf.h"
#include "featureTracking.h"
#include "commonFuncs.h"
#include "commonConfig.h"
#include "chi2.h"


using namespace std;
using namespace Eigen;

template <typename _Type>
bool outlierCheck(Matrix<_Type, Dynamic, 1> & r, Matrix<_Type, Dynamic, Dynamic> & cov);


template <typename _Type>
void Slide::copyFrom(const _Type & sld)
{
    R = sld.R;
    p = sld.p;
    frameID = sld.frameID;
    fej_p = sld.fej_p;
    fej_v = sld.fej_v;
}


void Slide::update(Matrix3f & new_R, Vector3f & new_p)
{
    R = new_R;
    p = new_p;
}


State::State()
{
}


void State::initializeWithGT(DataLoader & data, int init_frmID)
{
    auto frameIDs = data.getIMGInfo()->frameIDs;
    auto ptr = find_if(frameIDs.cbegin(), frameIDs.cend(), [init_frmID](auto frmID){return frmID == init_frmID;});
    if (ptr == frameIDs.cend())
    {
        cerr<<"can not initilize";
        exit(EXIT_FAILURE);
    }
    int lag = ptr - frameIDs.cbegin();
    double init_ts = *(data.getIMGInfo()->ts.begin() + lag);

    auto sub_gt = data.getGT()->getData(init_ts);

    first_block.R = quat2dcm(sub_gt.q.front());
    first_block.v = sub_gt.v.front();
    first_block.p = sub_gt.p.front();
    first_block.ba = sub_gt.ba.front();
    first_block.bg = sub_gt.bg.front();
    first_block.fej_p = sub_gt.p.front();
    first_block.fej_v = sub_gt.v.front();
    first_block.cov = 1e-4 * Matrix<float, 15, 15>::Identity();
    first_block.frameID = init_frmID;

    string im_path = data.getIMGInfo()->getPath(init_frmID);
    ImageArray im_array(im_path, init_frmID);
    // tracker.initialize(im_array);
    tracker.initializeWithORB(im_array);
}


void State::addSilde(const Slide & sld)
{
    slides.insert(slides.begin(), sld);
    slide_number++;
    int fbl = algo_cfg::FIRST_BLOCK_LEN;
    int dim = first_block.cov.rows();
    MatrixXf J1 = MatrixXf::Identity(dim, dim);
    MatrixXf J2 = MatrixXf::Zero(6, dim);
    MatrixXf eye3 = MatrixXf::Identity(3, 3);
    J2.block(0, 0, 3, 3) = eye3;
    J2.block(3, 6, 3, 3) = eye3;
    MatrixXf augJ(dim+6, dim);
    augJ << J1.topRows(fbl), 
            J2,
            J1.bottomRows(dim - fbl);
    first_block.cov = augJ * first_block.cov * augJ.transpose();
}


void State::deleteSlide()
{
    slides.pop_back();
    slide_number--;
    int fbl = algo_cfg::FIRST_BLOCK_LEN;
    int cov_len = fbl + slide_number * algo_cfg::SLIDE_SIZE;
    first_block.cov.conservativeResize(cov_len, cov_len);
}


void State::predict(IMU & imu_data, int curr_frmID)
{
    float g = sensor_cfg::GRAVITY;
    Vector3f gravity{0, 0, g};
    auto acc = imu_data.acc;
    auto gyro = imu_data.gyro;
    auto ts = imu_data.ts;
    Slide sld;
    sld.copyFrom(first_block);
    addSilde(sld);

    if (slide_number > algo_cfg::MAX_SLIDE_NUMBER)
        deleteSlide();

    MatrixXf covt = first_block.cov.transpose();    
    first_block.cov = 0.5 * (first_block.cov + covt);
    
    int fbl = algo_cfg::FIRST_BLOCK_LEN;
    auto R1 = first_block.R;
    auto v1 = first_block.v;
    auto p1 = first_block.p;
    auto ba = first_block.ba;
    auto bg = first_block.bg;
    auto fej_v1 = first_block.fej_v;
    auto fej_p1 = first_block.fej_p;

    int dim = first_block.cov.rows();
   
    MatrixXf cov11 = first_block.cov.block(0, 0, fbl, fbl);
    MatrixXf cov12 = first_block.cov.block(0, fbl, fbl, dim - fbl);

    auto imu_noise = sensor_cfg::IMU_NOISE;
    
    Vector3f w1, w2, a1, a2, w, a, rot_vec, v2, p2, tmp;
    Matrix3f dR, R2, jac_R_bg, jac_v_R, jac_v_ba, jac_p_R, jac_p_ba, jac_p_v;
    float dt;
    Matrix3f eye3 = Matrix3f::Identity();

    MatrixXf Qimu(fbl, fbl), phi(fbl, fbl), PHI(fbl, fbl);
    PHI.setIdentity();
    
    int imu_num = acc.size();

    for (int j0 = 1; j0 < imu_num;  j0++)
    {
        w1 = gyro.at(j0 - 1);
        w2 = gyro.at(j0);
        a1 = acc.at(j0 - 1);
        a2 = acc.at(j0);
        dt = ts.at(j0) - ts.at(j0 - 1);
        w = 0.5 * (w1 + w2) - bg;
        rot_vec = dt * w;
        dR = expSO3(rot_vec);
        a = 0.5 * (a1 - ba) + 0.5 * dR * (a2 - ba);
        R2 = R1 * dR;
        v2 = v1 + dt * (R1 * a - gravity);
        p2 = p1 + 0.5 * dt * (v1 + v2);
        jac_R_bg = -dt * R2 * rightApproxSO3(rot_vec);
        tmp =  v2 - fej_v1 + dt * gravity;
        jac_v_R = -vec2skewMat(tmp);
        jac_v_ba = -0.5 * dt * R1 * (eye3 + dR);
        tmp = p2 - fej_p1 - dt * fej_v1 + 0.5 * dt * dt / 2 * gravity;
        jac_p_R = -vec2skewMat(tmp);
        jac_p_v = dt * eye3;
        jac_p_ba = 0.5 * dt * jac_v_ba;
        phi.setIdentity();
        phi.block(0, 12, 3, 3) = jac_R_bg;
        phi.block(3, 0, 3, 3) = jac_v_R;
        phi.block(3, 9, 3, 3) = jac_v_ba;
        phi.block(6, 0, 3, 3) = jac_p_R;
        phi.block(6, 3, 3, 3) = jac_p_v;
        phi.block(6, 9, 3, 3) = jac_p_ba;
        Qimu = imuNoiseEvolveMat(imu_noise, dt, 2);
        cov11 = phi * cov11 * phi.transpose() + Qimu;
        PHI = phi * PHI;
        R1 = R2;
        v1 = v2;
        p1 = p2;
        fej_v1 = v2;
        fej_p1 = p2;
    }
    cov12 = PHI * cov12;
    first_block.cov.block(0, 0, fbl, fbl) = cov11;
    first_block.cov.block(0, fbl, fbl, dim - fbl) = cov12;
    first_block.cov.block(fbl, 0, dim - fbl, fbl) = cov12.transpose();
    first_block.R = R2;
    first_block.v = v2;
    first_block.p = p2;
    first_block.fej_v = v2;
    first_block.fej_p = p2;
    first_block.frameID = curr_frmID;
    //cout<<"pred_cov: "<<first_block.cov.diagonal()<<endl;
}


bool State::decideKeyFrame()
{
    if (0 == slide_number)
        return true;
    auto inverse_point_tracker_ptr = tracker.getInversePointTrackList();
    int curr_frmID = inverse_point_tracker_ptr->back().frameID;
    if (curr_frmID - first_block.frameID <= algo_cfg::KF_LEAST_FRAMES)
        return false;
    int prev_frmID = slides.front().frameID;
    
    auto iptr = find_if(inverse_point_tracker_ptr->cbegin(), inverse_point_tracker_ptr->cend(), 
                [prev_frmID](InversePointTrack iptk){return iptk.frameID == prev_frmID;});
    
    auto prev_pointIDs = iptr->pointIDs;
    auto curr_pointIDs = inverse_point_tracker_ptr->back().pointIDs;
    vector<int> co_pids;

    sort(prev_pointIDs.begin(), prev_pointIDs.end());
    sort(curr_pointIDs.begin(), curr_pointIDs.end());
    set_intersection(prev_pointIDs.begin(), prev_pointIDs.end(), curr_pointIDs.begin(), curr_pointIDs.end(), 
            back_inserter(co_pids));
    if (co_pids.size() < algo_cfg::COVISION)
        return true;
    
    float nps = 0;
    int cnt = 0;
    for (auto ptk : *tracker.getPointTrackList())
    {
        auto pid_ptr = find_if(co_pids.cbegin(), co_pids.cend(), [ptk](int pid){return pid == ptk.pointID;});
        if (pid_ptr == co_pids.cend())
            continue;
        co_pids.erase(pid_ptr);
        auto curr_pts = ptk.points.back();
        auto prev_pts =  *(ptk.points.cend() - (curr_frmID - prev_frmID + 1));
        nps += norm(curr_pts - prev_pts);
        cnt++;
    }
    float mdist = nps / cnt;
    if (mdist > algo_cfg::PARRALAX)
        return true;
    return false;
}


void State::deletePointTracks()
{
    auto point_track_list_ptr = tracker.getPointTrackList();
    for (auto pitr = point_track_list_ptr->cbegin(); pitr != point_track_list_ptr->cend(); pitr++)
    {
        if (slide_number > algo_cfg::DIST_TO_CURRENT)
        {
            if (pitr->frameIDs.back() <= slides.at(algo_cfg::DIST_TO_CURRENT).frameID)
            {
                point_track_list_ptr->erase(pitr);
                pitr--;
                continue;
            }
            if (pitr->frameIDs.back() != first_block.frameID && pitr->frameIDs.size() <= algo_cfg::TRACK_LENGTH)
            {
                point_track_list_ptr->erase(pitr);
                pitr--;
            }
        }
    }
}


void State::deleteLostPointTracks()
{
    auto point_track_list_ptr = tracker.getPointTrackList();
    for (auto pitr = point_track_list_ptr->cbegin(); pitr != point_track_list_ptr->cend(); pitr++)
    {
        if (pitr->frameIDs.back() < first_block.frameID)
        {
            point_track_list_ptr->erase(pitr);
            pitr--;
        }
    }
}


void State::deleteInversePointTracks()
{
    if (slide_number >= algo_cfg::MAX_SLIDE_NUMBER)
    {
        int dist_frmID = slides.back().frameID;
        for (auto iptr = tracker.getInversePointTrackList()->begin(); iptr != tracker.getInversePointTrackList()->end(); iptr++)
        {
            if (iptr->frameID < dist_frmID)
            {
                tracker.getInversePointTrackList()->erase(iptr);
                iptr--;
            }
        }
    }
}


void State::triangulation()
{
    vector<int> sld_frmIDs;
    for (auto sld : slides)
    {
        sld_frmIDs.insert(sld_frmIDs.cbegin(), sld.frameID);
    }

    Matrix3f Rb2c;
    cv::cv2eigen(sensor_cfg::ROTATION_MATRIX_B2C, Rb2c);
    Vector3f Pb2c;
    cv::cv2eigen(sensor_cfg::POSITION_B2C, Pb2c);
    Matrix3f K;
    cv::cv2eigen(sensor_cfg::CAM_INTRINSIC_MAT, K);

    auto point_track_list = tracker.getPointTrackList();
    for (auto ptk_itr = point_track_list->begin(); ptk_itr != point_track_list->end(); ptk_itr++)
    {
        if (ptk_itr->xyz.tri_error  > 0)
            continue;

        if (ptk_itr->frameIDs.size() < imgproc_cfg::KF_NUMBER)
            continue;
         
        if (slides.size() < imgproc_cfg::KF_NUMBER)
            continue;

        auto track_points = ptk_itr->points;
        auto track_frmIDs = ptk_itr->frameIDs;
        vector<int> inter_frmIDs;

        set_intersection(track_frmIDs.begin(), track_frmIDs.end(), sld_frmIDs.begin(), sld_frmIDs.end(), back_inserter(inter_frmIDs));
        int obs_num = inter_frmIDs.size();
    
        if (obs_num < imgproc_cfg::KF_NUMBER)
            continue;  
        
        if (track_frmIDs.back() == first_block.frameID)
        {
            if (obs_num < slide_number)
            {
                continue;
            }
        }
        
        MatrixXf A(2 * obs_num, 4);
        Matrix<float, 3, 4> P;
        vector< Matrix<float, 3, 4> > Pc;
        vector< Matrix<float, 2, 1> > uvs;
        VectorXf error(obs_num);
        Matrix3f Rc2w;
        Vector3f Pc2w;
        Matrix<float, 2, 1> p2d;
        cv::Point2f cv_p2d;
        Vector3f p3d;
        int idx = 0;

        for (int frmID : inter_frmIDs)
        {
            auto sld_itr = find_if(slides.cbegin(), slides.cend(), [frmID](auto sld){return sld.frameID == frmID;});
            Rc2w = sld_itr->R * Rb2c.transpose();
            Pc2w = sld_itr->p - Rc2w * Pb2c;
            auto tck_itr = find_if(track_frmIDs.cbegin(), track_frmIDs.cend(), [frmID](auto tck_frmID){return tck_frmID == frmID;});
            cv_p2d = track_points.at(tck_itr - track_frmIDs.cbegin());
            p2d(0) = cv_p2d.x;
            p2d(1) = cv_p2d.y;
            uvs.push_back(p2d);
            Matrix3f tmp = K * Rc2w.transpose();
            P.block(0, 0, 3, 3) = tmp;
            P.block(0, 3, 3, 1) = -tmp * Pc2w;
            Pc.push_back(P);
            A.row(2*idx) = cv_p2d.x * P.row(2) - P.row(0);
            A.row(2*idx + 1) = cv_p2d.y * P.row(2) - P.row(1);
            idx++;
        }
        
        JacobiSVD< MatrixXf > svd(A, ComputeThinV);
        
        Matrix<float, 4, 1> w = svd.matrixV().col(3);
        p3d = w.head(3) / w(3);
        Matrix<float, 4, 1> h_p3d;
        h_p3d.head(3) = p3d;
        h_p3d(3) = 1;
        
        bool depth_check = true;
        for (auto itr = Pc.cbegin(); itr != Pc.cend(); itr ++)
        {
            Vector3f p2d_proj = (*itr) * h_p3d; 
            if (p2d_proj(2) < imgproc_cfg::MIN_DEPTH || p2d_proj(2) > imgproc_cfg::MAX_DEPTH)
            {
                depth_check = false;
                break;
            }
            Vector2f h_p2d_proj = p2d_proj.head(2) / p2d_proj(2);
            int lag = itr - Pc.cbegin();
            error(lag) = (h_p2d_proj - uvs.at(lag)).norm();
        }
        
        if (depth_check == false)
            continue;
            
        float m_err = error.mean();
        if (m_err > imgproc_cfg::REPROJECTION_ERROR)
            continue;
 
        ptk_itr->xyz.update(p3d, p3d, m_err, true);
    }
}


void State::projection(ProjectionInformation & proj_info)
{
    Matrix3f Rb2c;
    Vector3f Pb2c;
    Matrix3f K;
    cv::cv2eigen(sensor_cfg::ROTATION_MATRIX_B2C, Rb2c);
    cv::cv2eigen(sensor_cfg::POSITION_B2C, Pb2c);
    cv::cv2eigen(sensor_cfg::CAM_INTRINSIC_MAT, K);
    float img_noise2 = sensor_cfg::IMG_NOISE * sensor_cfg::IMG_NOISE;

    vector<int> state_frmIDs;
    for (auto sld : slides)
        state_frmIDs.insert(state_frmIDs.cbegin(), sld.frameID);
    state_frmIDs.push_back(first_block.frameID);

    int tot_obs_num = 0;
    int inner_point_num = 0;
    auto point_tracks = *tracker.getPointTrackList();
    //unsigned seed = chrono::system_clock::now().time_since_epoch().count(); 
    //shuffle(point_tracks.begin(), point_tracks.end(), std::default_random_engine(seed));  
    for (auto ptk : point_tracks)
    {
        if (inner_point_num > 300)
            break;
        
        if (ptk.xyz.tri_error < 0)
            continue;
        
        auto track_frmIDs = ptk.frameIDs;
        vector<int> inter_frmIDs;

        set_intersection(state_frmIDs.cbegin(), state_frmIDs.cend(), track_frmIDs.cbegin(), track_frmIDs.cend(),
            back_inserter(inter_frmIDs));
        
        if (inter_frmIDs.size() < 3)
            continue;
        
        int fbl = algo_cfg::FIRST_BLOCK_LEN;
        Vector3f p3d = ptk.xyz.position;
        Vector3f fej_p3d = ptk.xyz.fej_position;

        int total_col_nums = first_block.cov.cols();

        Matrix3f Rb2w, Rc2w;
        Vector3f Pb2w, fej_p, Pc2w, h_pre_z;
        Vector2f obs_z, pre_z, err;
        Matrix<float, 2, 3> jac_z;
        VectorXf ri(2 * inter_frmIDs.size());
        MatrixXf Jxi(2 * inter_frmIDs.size(), total_col_nums);
        MatrixXf Jfi(2 * inter_frmIDs.size(), 3);
        ri.setZero();
        Jxi.setZero();
        Jfi.setZero();
        int cr, cp;
        int r_idx = 0;
        for (int frmID : inter_frmIDs)
        {
            if (frmID == first_block.frameID)
            {
                Rb2w = first_block.R;
                Pb2w = first_block.p;
                fej_p = first_block.fej_p;
                cr = 0;
                cp = 6;
            }
            else
            {
                auto sld_itr = find_if(slides.cbegin(), slides.cend(), [frmID](auto sld){return sld.frameID == frmID;});
                int lag = sld_itr - slides.cbegin();
                Rb2w = sld_itr->R;
                Pb2w = sld_itr->p;
                fej_p = sld_itr->fej_p;
                cr = fbl + lag * algo_cfg::SLIDE_SIZE;
                cp = cr + 3;
            }
            auto tck_itr = find_if(track_frmIDs.cbegin(), track_frmIDs.cend(), [frmID](int tck_frmID){return tck_frmID == frmID;});
            Point2f cv_obs_z = ptk.points.at(tck_itr - track_frmIDs.cbegin());

            obs_z(0) = cv_obs_z.x;
            obs_z(1) = cv_obs_z.y;

            Rc2w = Rb2w * Rb2c.transpose();
            Pc2w = Pb2w - Rc2w * Pb2c;
            h_pre_z = K * Rc2w.transpose() * (p3d - Pc2w); 
            invHomo2dAndJac(h_pre_z, pre_z, jac_z);
            err = obs_z - pre_z;

            ri.middleRows(2 * r_idx, 2) = err;
            Matrix<float, 2, 3> tmp = jac_z * K * Rc2w.transpose();
            Vector3f p_dist = fej_p3d - fej_p;
            Jxi.block(2 * r_idx, cr, 2, 3) = tmp * vec2skewMat(p_dist);
            Jxi.block(2 * r_idx, cp, 2, 3) = -tmp;
            Jfi.middleRows(2 * r_idx, 2) = tmp;  
            r_idx++;   
        }

        JacobiSVD< MatrixXf > svd0(Jfi.transpose(), ComputeFullV);
        MatrixXf null_mat = svd0.matrixV().rightCols(Jfi.rows() - 3);

        MatrixXf null_Jxi = null_mat.transpose() * Jxi;
        VectorXf null_ri = null_mat.transpose() * ri;
        MatrixXf ri_noise_mat(null_Jxi.rows(), null_Jxi.rows());
        ri_noise_mat.setIdentity();
        ri_noise_mat *= img_noise2;
        MatrixXf ri_cov = null_Jxi * first_block.cov * null_Jxi.transpose() + ri_noise_mat;

        if (outlierCheck(null_ri, ri_cov) == false)
        {
            inner_point_num++;
            float wt = 1 / sensor_cfg::IMG_NOISE;
            null_Jxi *= wt;
            null_ri *= wt;
            int add_obs_num = null_Jxi.rows();
            proj_info.Jx.conservativeResize(tot_obs_num + add_obs_num, first_block.cov.cols());
            proj_info.error.conservativeResize(tot_obs_num + add_obs_num, 1);
            proj_info.Jx.bottomRows(add_obs_num) = null_Jxi;
            proj_info.error.bottomRows(add_obs_num) = null_ri;
            tot_obs_num += add_obs_num;
        }
    }
    cout<<"inner_point_num: "<<inner_point_num<<endl;
}


void State::visualCorrection()
{
    ProjectionInformation proj_info;
    projection(proj_info);

    if (proj_info.error.rows() > 0)
    {
        MatrixXf Jx = proj_info.Jx;
        VectorXf error = proj_info.error;
        if (algo_cfg::DO_QR)
        {
            if (proj_info.Jx.rows() > proj_info.Jx.cols())
            {
                HouseholderQR<MatrixXf> qr(proj_info.Jx);
                MatrixXf R = qr.matrixQR().triangularView<Upper>();
                MatrixXf Q = qr.householderQ();
                VectorXf rowsum_R = R.array().abs().rowwise().sum();
                int idx = findFirst(rowsum_R);
                Jx = R.topRows(idx);
                MatrixXf Q_reduce = Q.leftCols(idx);
                error = Q_reduce.transpose() * proj_info.error;
            }
        }

        float img_noise2 = sensor_cfg::IMG_NOISE * sensor_cfg::IMG_NOISE;
        MatrixXf obs_cov(Jx.rows(), Jx.rows());
        obs_cov.setIdentity();
        obs_cov *= img_noise2;
    
        MatrixXf gain_cov = Jx * first_block.cov * Jx.transpose() + obs_cov;
        MatrixXf kal = first_block.cov * Jx.transpose() * gain_cov.inverse();
        VectorXf dx = kal * error;
        
        //cout<<"dx:"<<dx<<endl;

        Vector3f dR_vec = dx.middleRows(0, 3);
        first_block.R = expSO3(dR_vec) * first_block.R;
        first_block.v += dx.middleRows(3, 3);
        first_block.p += dx.middleRows(6, 3);
        first_block.ba += dx.middleRows(9, 3);
        first_block.bg += dx.middleRows(12, 3);
        int fbl = algo_cfg::FIRST_BLOCK_LEN;
        for (auto sld_itr = slides.begin(); sld_itr != slides.end(); sld_itr++)
        {
            int lag = sld_itr - slides.begin();
            int r_idx = fbl + lag * algo_cfg::SLIDE_SIZE;
            dR_vec = dx.middleRows(r_idx, 3);
            sld_itr->R = expSO3(dR_vec) * sld_itr->R;
            sld_itr->p += dx.middleRows(r_idx + 3, 3);
        }
        MatrixXf eyex(first_block.cov.rows(), first_block.cov.cols());
        eyex.setIdentity();
        MatrixXf tmp_M = eyex - kal * Jx;
        first_block.cov = tmp_M * first_block.cov;
        //first_block.cov = tmp_M * first_block.cov * tmp_M.transpose();
        //first_block.cov += kal * obs_cov * kal.transpose();
        //cout<<"upd_cov: "<<first_block.cov.diagonal()<<endl;
    }  
}


State::~State()
{
}


template <typename _Type>
bool outlierCheck(Matrix<_Type, Dynamic, 1> & r, Matrix<_Type, Dynamic, Dynamic> & cov)
{
    int dof = r.rows();
    float v = r.transpose() * cov.inverse() * r;
    float chi2_95 = chi2_fast_95::chi2_quatile95(dof);
    float reproj_err = r.transpose() * r;
    reproj_err /= float(dof);
    return (v > chi2_95 || reproj_err > 2);
}











