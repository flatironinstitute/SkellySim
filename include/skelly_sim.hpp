#ifndef SKELLYSIM_HPP
#define SKELLYSIM_HPP

#ifndef GIT_TAG
#define GIT_TAG "<undefined tag>"
#endif

#ifndef GIT_COMMIT
#define GIT_COMMIT "<undefined commit>"
#endif

#include <vector>

#include <toml.hpp>

#include <msgpack.hpp>
#define EIGEN_MATRIX_PLUGIN "eigen_matrix_plugin.h"
#define EIGEN_QUATERNION_PLUGIN "eigen_quaternion_plugin.h"

#include <Eigen/Core>

typedef Eigen::Map<Eigen::VectorXd> VectorMap;
typedef Eigen::Map<const Eigen::VectorXd> CVectorMap;
typedef Eigen::Map<Eigen::ArrayXd> ArrayMap;
typedef Eigen::Map<const Eigen::ArrayXd> CArrayMap;
typedef Eigen::Map<Eigen::MatrixXd> MatrixMap;
typedef Eigen::Map<const Eigen::MatrixXd> CMatrixMap;
typedef const Eigen::Ref<const Eigen::ArrayXd> ArrayRef;
typedef const Eigen::Ref<const Eigen::VectorXd> VectorRef;
typedef const Eigen::Ref<const Eigen::MatrixXd> MatrixRef;

/// Struct of parameters for exponentially decaying fiber-periphery interaction
typedef struct {
    double f_0; ///< strength of interaction
    double l_0; ///< characteristic length of interaction
} fiber_periphery_interaction_t;

typedef struct {
    bool active;              ///< If periphery binding enabled
    double polar_angle_start; ///< Angle from z axis where binding starts
    double polar_angle_end;   ///< Angle from z axis where binding ends
    double threshold;         ///< Minimum distance from cortex where binding will begin
} periphery_binding_t;

template <class T>
struct ActiveIterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type *;
    using reference = value_type &;

    ActiveIterator(int index, std::vector<T> &objs) : m_index(index), container_ref(objs) {
        while (m_index < container_ref.size() && !container_ref[m_index].active()) {
            m_index++;
        }
    }

    reference operator*() const { return container_ref[m_index]; }
    pointer operator->() { return &container_ref[m_index]; }

    // Prefix increment
    ActiveIterator &operator++() {
        do {
            m_index++;
        } while (m_index < container_ref.size() && !container_ref[m_index].active());

        return *this;
    }

    // Postfix increment
    ActiveIterator operator++(int) {
        ActiveIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const ActiveIterator &a, const ActiveIterator &b) { return a.m_index == b.m_index; };
    friend bool operator!=(const ActiveIterator &a, const ActiveIterator &b) { return a.m_index != b.m_index; };

  private:
    int m_index;
    std::vector<T> &container_ref;
};

#endif
