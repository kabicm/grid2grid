#include <vector>
#include <complex>

namespace grid2grid {
using zfloat = std::complex<float>;
using zdouble = std::complex<double>;

template <typename Scalar>
std::vector<Scalar>& get_global_pool() {
    static std::vector<Scalar> pool;
    return pool;
}

};


template
std::vector<float>& grid2grid::get_global_pool<float>();
template
std::vector<double>& grid2grid::get_global_pool<double>();
template
std::vector<grid2grid::zfloat>& grid2grid::get_global_pool<grid2grid::zfloat>();
template
std::vector<grid2grid::zdouble>& grid2grid::get_global_pool<grid2grid::zdouble>();
