#include <vector>
#include <algorithm>

#ifndef _CHI2_H
#define _CHI2_H

namespace chi2_fast_95
{
    
    const static std::array<int, 15> INDEX{1, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960};
    const static std::array<float, 30> PARAMS{1.69101006, 2.67256959, 1.42827388, 4.05222026, 1.30293086, 5.40961364, 
                                           1.21418901, 7.28669559, 1.15144756, 9.90941434, 1.10708881, 13.59578217, 
                                           1.07572373, 18.7931972, 1.05354521, 26.13238638, 1.03786242, 36.5038631, 
                                           1.02677288, 51.16597064, 1.01893132, 71.89756134, 1.01338648, 101.21382064, 
                                           1.00946568, 142.67141864, 1.00669325, 201.30001128, 1.00473284, 284.2124426};

    float chi2_quatile95(int dof)
    {
        auto itr = find_if(INDEX.cbegin(), INDEX.cend(), [dof](int n){return n <= dof;});
        int p_idx = itr - INDEX.cbegin();
        float a = PARAMS[2 * p_idx];
        float b = PARAMS[2 * p_idx + 1];
        float chi2_95 = a * dof + b;
        return chi2_95;
    }

}





#endif
