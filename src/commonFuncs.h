#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <iostream>
#include<cmath>

using namespace Eigen;
using namespace std;

#ifndef COMMONFUNCS_H_
#define COMMONFUNCS_H_


template <typename _Type> 
_Type signLocalVersion(_Type x)
{
    return (_Type)(x > 0) - (_Type)(x < 0);
}


template <typename _Type>
Matrix<_Type, 3, 3> quat2dcm(Quaternion<_Type> & q)
{
    q.normalize();
    _Type q0 = q.w();
    _Type q1 = q.x();
    _Type q2 = q.y();
    _Type q3 = q.z();
    cout<<q0<<"\t"<<q1<<"\t"<<q2<<"\t"<<q3<<endl;
    _Type q01 = q0 * q1;
    _Type q02 = q0 * q2;
    _Type q03 = q0 * q3;
    _Type q11 = q1 * q1;
    _Type q12 = q1 * q2;
    _Type q13 = q1 * q3;
    _Type q22 = q2 * q2;
    _Type q23 = q2 * q3;
    _Type q33 = q3 * q3;

    Matrix<_Type, 3, 3> R;
    R(0, 0) = 1 - 2 * (q22 + q33);
    R(0, 1) = 2 * (q12 - q03);
    R(0, 2) = 2 * (q13 + q02);

    R(1, 0) = 2 * (q12 + q03);
    R(1, 1) = 1 - 2 * (q11 + q33);
    R(1, 2) = 2 * (q23 - q01);

    R(2, 0) = 2 * (q13 - q02);
    R(2, 1) = 2 * (q23 + q01);
    R(2, 2) = 1 - 2 * (q11 + q22);
    return R;
}


template <typename _Type>
Matrix<_Type, 3, 3> quat2dcm(Matrix<_Type, 4, 1> & q, bool w_first = true)
{
    q.normalize();
    _Type q0, q1, q2, q3;
    q1 = q(1);
    q2 = q(2);
    if (w_first)
    {
        q0 = q(0);
        q3 = q(3);
    }
    else
    {
        q0 = q(3);
        q3 = q(0);
    }
    _Type q01 = q0 * q1;
    _Type q02 = q0 * q2;
    _Type q03 = q0 * q3;
    _Type q11 = q1 * q1;
    _Type q12 = q1 * q2;
    _Type q13 = q1 * q3;
    _Type q22 = q2 * q2;
    _Type q23 = q2 * q3;
    _Type q33 = q3 * q3;

    Matrix<_Type, 3, 3> R;
    R(0, 0) = 1 - 2 * (q22 + q33);
    R(0, 1) = 2 * (q12 - q03);
    R(0, 2) = 2 * (q13 + q02);

    R(1, 0) = 2 * (q12 + q03);
    R(1, 1) = 1 - 2 * (q11 + q33);
    R(1, 2) = 2 * (q23 - q01);

    R(2, 0) = 2 * (q13 - q02);
    R(2, 1) = 2 * (q23 + q01);
    R(2, 2) = 1 - 2 * (q11 + q22);
    return R;
}


template <typename _Type>
Matrix<_Type, 2, 1> acc2ang(const Matrix<_Type, 3, 1> & acc)
{
    acc.normalize();
    _Type ax = acc(0);
    _Type ay = acc(1);
    _Type az = acc(2);
    _Type theta = asin(-ax);
    _Type phi = atan2(ay, az);
    Matrix<_Type, 2, 1> ang;
    ang << theta, phi;
    return ang;
}


template <typename _Type>
Matrix<_Type, 3, 3> ang2dcm(const _Type psi, const _Type theta, const _Type phi)
{
    _Type s_psi = sin(psi);
    _Type c_psi = cos(psi);
    _Type s_theta = sin(theta);
    _Type c_theta = cos(theta);
    _Type s_phi = sin(phi);
    _Type c_phi = cos(phi);
    Matrix<_Type, 3, 3> R;
    R(0, 0) = c_theta * c_psi;
    R(0, 1) = s_phi * s_theta * c_psi - c_phi * s_psi;
    R(0, 2) = c_phi * s_theta * c_psi + s_phi * s_psi;

    R(1, 0) = c_theta * s_psi;
    R(1, 1) = s_phi * s_theta * s_psi + c_phi * c_psi;
    R(1, 2) = c_phi * s_theta * s_psi - s_phi * c_psi;

    R(2, 0) = -s_theta;
    R(2, 1) = s_phi * c_theta;
    R(2, 2) = c_phi * c_theta;
    return R;
}


template <typename _Type>
Matrix<_Type, 3, 3> ang2dcm(const Matrix<_Type, 3, 1> & ang)
{
    _Type psi = ang(0);
    _Type theta = ang(1);
    _Type phi = ang(2);
    return ang2dcm(psi, theta, phi);
}


template <typename _Type>
Matrix<_Type, 3, 1> dcm2ang(const Matrix<_Type, 3, 3> & R)
{
    _Type ct2 = sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2));
    _Type psi, theta, phi;
    if (ct2 > 1e-4)
    {
        psi = atan2(R(1, 0), R(0, 0));
        theta = asin(-R(2, 0));
        phi = atan2(R(2, 1), R(2, 2));
    }
    else
    {
        psi = 0.0;
        theta = asin(-R(2, 0));
        phi = atan2(-R(1, 2), R(1, 1));
    }
    Matrix<_Type, 3, 1> ang;
    ang << psi, theta, phi;
    return ang;
}


template <typename _Type>
Matrix<_Type, 3, 3> vec2skewMat(const Matrix<_Type, 3, 1> & vec)
{
    Matrix<_Type, 3, 3> M = Matrix<_Type, 3, 3>::Zero();
    M(0, 1) = -vec(2);
    M(0, 2) = vec(1);
    M(1, 0) = vec(2);
    M(1, 2) = -vec(0);
    M(2, 0) = -vec(1);
    M(2, 1) = vec(0);
    return M;
}


template <typename _Type>
Matrix<_Type, 3, 1> skewMat2vec(const Matrix<_Type, 3, 3> & M)
{
    Matrix<_Type, 3, 1> v;
    v(0) = 0.5 * (M(2, 1) - M(1, 2));
    v(1) = 0.5 * (M(0, 2) - M(2, 0));
    v(2) = 0.5 * (M(1, 0) - M(0, 1));
    return v;
}


template <typename _Type>
Matrix<_Type, Dynamic, Dynamic> perfectRot(const Matrix<_Type, Dynamic, Dynamic> & M)
{
    Matrix<_Type, Dynamic, Dynamic> R;
    JacobiSVD< Matrix<_Type, Dynamic, Dynamic> > svd(M, ComputeThinU | ComputeThinV);
    Matrix<_Type, Dynamic, Dynamic> sig_mat = Matrix<_Type, Dynamic, Dynamic>(svd.singularValues().asDiagonal());
    R = svd.matrixU() * sig_mat * svd.matrixV().transpose();
    return R;
}


template <typename _Type>
Matrix<_Type, 3, 3> perfectRot(const Matrix<_Type, 3, 3> & M)
{
    cout<<"M"<<M<<endl;
    Matrix<_Type, 3, 3> R;
    JacobiSVD< Matrix<_Type, Dynamic, Dynamic> > svd(M, ComputeThinU | ComputeThinV);
    Matrix<_Type, 3, 3> sig_mat = Matrix<_Type, 3, 3>(svd.singularValues().asDiagonal());
    R = svd.matrixU() * sig_mat * svd.matrixV().transpose();
    return R;
}


template <typename _Type>
Matrix<_Type, 3, 3> expSO3(const Matrix<_Type, 3, 1> & v)
{
    _Type n = v.norm();
    Matrix<_Type, 3, 3> R;
    Matrix<_Type, 3, 3> eye3 = Matrix<_Type, 3, 3>::Identity();
    if (n < 1.0e-4)
    {
        auto vx = vec2skewMat(v);
        R = eye3 + vx + 0.5 * vx * vx;
    }
    else
    {
        Matrix<_Type, 3, 1> a = v / n;
        auto ax = vec2skewMat(a);
        R = eye3 + sin(n) * ax + (1 - cos(n)) * ax * ax;
    }
    return R;
}


template <typename _Type>
Matrix<_Type, 3, 1> logSO3(const Matrix<_Type, 3, 3> & R)
{
    _Type x = 0.5 * (R.trace()-1);
    Matrix<_Type, 3, 1> v;
    Matrix<_Type, 3, 3> eye3 = Matrix<_Type, 3, 3>::Identity();
    Matrix<_Type, 3, 3> M;
    if (abs(x + 1) < 1e-3)
    {
        M = 0.5 * (R + eye3);
        v(0) = sqrt(M(0, 0));
        v(1) = signLocalVersion(M(0, 1)) * sqrt(M(1, 1));
        v(2) = signLocalVersion(M(0, 2)) * sqrt(M(2, 2));
        v *= M_PI;
    }
    else
    {
        x = x * (abs(x) <= 1) + (x > 1) - (x < -1);
        _Type theta = acos(x);
        if (theta < 1e-4)
            M = 0.5 * (R - R.transpose());
        else
            M = 0.5 * theta / sin(theta) * (R - R.transpose());
        v = skewMat2vec(M);
    }
    return v;
}
        

template <typename _Type>
Matrix<_Type, 3, 3> rightApproxSO3(const Matrix<_Type, 3, 1> & v)
{
    _Type n = v.norm();
    Matrix<_Type, 3, 3> eye3 = Matrix<_Type, 3, 3>::Identity();
    Matrix<_Type, 3, 3> Jr;
    if (n < 1e-4)
    {
        auto vx = vec2skewMat(v);
        Jr = eye3 - 0.5 * vx + 1/6 * vx * vx;
    }
    else
    {
        Matrix<_Type, 3, 1> a = v / n;
        auto ax = vec2skewMat(a);
        _Type x2 = 1 - sin(n) / n;
        _Type x1 = (1 - cos(n)) / n;
        Jr = eye3 - x1 * ax + x2 * ax * ax;
    }
    return Jr;
}


template <typename _Type>
Matrix<_Type, 15, 15> imuNoiseEvolveMat(const array<_Type, 4> & imu_noise, _Type dt, int order=1)
{
    _Type na2 = imu_noise[0] * imu_noise[0];
    _Type ng2 = imu_noise[1] * imu_noise[1];
    _Type nba2 = imu_noise[2] * imu_noise[2];
    _Type nbg2 = imu_noise[3] * imu_noise[3];
    Matrix<_Type, 15, 15> Qimu;
    Qimu.setZero();
    _Type dt2 = dt * dt;
    _Type dt3 = dt2 * dt;

    Matrix<_Type, 3, 3> eye3 =  Matrix<_Type, 3, 3>::Identity();
    Qimu.block(0, 0, 3, 3) = dt * ng2 * eye3;
    Qimu.block(3, 3, 3, 3) = dt * na2 * eye3;
    Qimu.block(9, 9, 3, 3) = dt * nba2 * eye3;
    Qimu.block(12, 12, 3, 3) = dt * nbg2 * eye3;

    if (order == 2)
    {
        Qimu.block(6, 6, 3, 3) = dt3 / 3 * na2 * eye3;
        Qimu.block(3, 6, 3, 3) = 0.5 * dt2 * na2 * eye3;
        Qimu.block(6, 3, 3, 3) = 0.5 * dt2 * na2 * eye3;
    }
    return Qimu;
}


template <typename _Type>
void homo3d(Matrix<_Type, 3, 1> & vec, Matrix<_Type, 4, 1> & hvec)
{
    hvec.head(3) = vec;
    hvec(3) = 1;
}


template <typename _Type>
void invHomo3d(Matrix<_Type, 4, 1> & hvec, Matrix<_Type, 3, 1> & vec)
{
    vec = hvec.head(3) / hvec(3);
}


template <typename _Type>
void homo2d(Matrix<_Type, 2, 1> & vec, Matrix<_Type, 3, 1> & hvec)
{
    hvec.head(2) = vec;
    hvec(2) = 1;
}


template <typename _Type>
void invHomo2d(Matrix<_Type, 3, 1> & hvec, Matrix<_Type, 2, 1> & vec)
{
    vec = hvec.head(2) / hvec(2);
}


template <typename _Type>
void invHomo2dAndJac(Matrix<_Type, 3, 1> & hvec, Matrix<_Type, 2, 1> & vec, Matrix<_Type, 2, 3> & Jac)
{
    _Type rho = 1 / hvec(2);
    vec = rho * hvec.head(2);
    Jac.setZero();
    Jac(0, 0) = 1;
    Jac(1, 1) = 1;
    Jac(0, 2) = -vec(0);
    Jac(1, 2) = -vec(1);
}

template <typename _Type>
int findFirst(Matrix<_Type, Dynamic, 1> & vec)
{
    int idx;
    for (idx = 0; idx < vec.rows(); idx++)
    {
        if (vec(idx) < 1e-6)
            break;
    }
    return idx;
}


#endif


