#ifndef PARSE_UTIL_HPP
#define PARSE_UTIL_HPP

#include <skelly_sim.hpp>

#include <Eigen/Geometry>
#include <iostream>

namespace parse_util {

template <typename T = Eigen::VectorXd>
inline T convert_array(const toml::array &src) {
    if constexpr (std::is_same_v<T, Eigen::Quaterniond>) {
        double tmp[4];
        for (size_t i = 0; i < src.size(); ++i)
            tmp[i] = src.at(i).as_floating();

        return Eigen::Quaterniond(tmp);
    }
    else if constexpr (std::is_same_v<T, Eigen::Vector3i>) {
        T trg;
        for (size_t i = 0; i < 3; ++i)
            trg[i] = src.at(i).as_integer();

        return trg;
    }
    else {
        T trg(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            trg[i] = src.at(i).as_floating();

        return trg;
    }
}

} // namespace parse_util
#endif
