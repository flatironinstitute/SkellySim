#ifndef SKELLYSIM_HPP
#define SKELLYSIM_HPP

#ifndef GIT_TAG
#define GIT_TAG "<undefined tag>"
#endif

#ifndef GIT_COMMIT
#define GIT_COMMIT "<undefined commit>"
#endif

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
typedef struct  {
    double f_0; ///< strength of interaction
    double lambda; ///< characteristic length of interaction
} fiber_periphery_interaction_t;

#endif
