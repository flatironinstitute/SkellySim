#include <skelly_sim.hpp>

#include <body.hpp>
#include <fiber_container_finite_difference.hpp>
#include <kernels.hpp>
#include <system.hpp>
#include <tuple>
#include <utils.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/// @brief Construct body from relevant toml config and system params
///
///   @param[in] body_table toml table from pre-parsed config
///   @param[in] params Pre-constructed Params object
///   surface).
///   @return Body object that has been appropriately rotated. Other internal cache variables are _not_ updated.
/// @see update_cache_variables
Body::Body(const toml::value &body_table, const Params &params) {}

const std::string Body::EXTFORCE_name[] = {"Linear", "Oscillatory"};

