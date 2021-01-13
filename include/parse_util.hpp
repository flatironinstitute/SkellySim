#ifndef PARSE_UTIL_HPP
#define PARSE_UTIL_HPP

#include <toml.hpp>
#include <Eigen/Core>
namespace parse_util {

inline Eigen::VectorXd convert_array(const toml::array *src) {
    Eigen::VectorXd trg(src->size());
    for (size_t i = 0; i < src->size(); ++i)
        trg[i] = src->get_as<double>(i)->get();

    return trg;
}

inline Eigen::VectorXd parse_array_key(const toml::table *tbl, const std::string &key) {
    const toml::node *src_node = tbl->get(key);
    if (!src_node)
        throw std::runtime_error("Key not found \"" + key + "\"");
    const toml::array *src = src_node->as_array();

    return convert_array(src);
}

template <typename T>
T parse_val_key(const toml::table *tbl, const std::string &key) {
    const toml::node *src_node = tbl->get(key);
    if (!src_node)
        throw std::runtime_error("Key not found \"" + key + "\"");

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
