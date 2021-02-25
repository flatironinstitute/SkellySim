#ifndef PARSE_UTIL_HPP
#define PARSE_UTIL_HPP

#include <skelly_sim.hpp>

#include <Eigen/Geometry>
#include <iostream>

namespace parse_util {

template <typename T = Eigen::VectorXd>
inline T convert_array(const toml::array *src) {
    if constexpr (std::is_same_v<T, Eigen::Quaterniond>) {
        double tmp[4];
        for (size_t i = 0; i < src->size(); ++i)
            tmp[i] = src->get_as<double>(i)->get();

        return Eigen::Quaterniond(tmp);
    } else {
        T trg(src->size());
        for (size_t i = 0; i < src->size(); ++i)
            trg[i] = src->get_as<double>(i)->get();

        return trg;
    }
}

template <typename T = Eigen::VectorXd>
inline T parse_array_key(const toml::table *tbl, const std::string &key) {
    const toml::node *src_node = tbl->get(key);
    if (!src_node) {
        std::cerr << "\n\n" << *tbl << "\n\n";
        throw std::runtime_error("Key not found \"" + key + "\"");
    }
    const toml::array *src = src_node->as_array();

    return convert_array<T>(src);
}

template <typename T>
T parse_val_key(const toml::table *tbl, const std::string &key) {
    const toml::node *src_node = tbl->get(key);
    if (!src_node) {
        std::cerr << "\n\n" << *tbl << "\n\n";
        throw std::runtime_error("Key not found \"" + key + "\"");
    }

    return src_node->as<T>()->get();
}

template <typename T>
T parse_val_key(const toml::table *tbl, const std::string &key, T default_val) {
    const toml::node *src_node = tbl->get(key);
    if (!src_node)
        return default_val;

    return src_node->as<T>()->get();
}

} // namespace parse_util
#endif
