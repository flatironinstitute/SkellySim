#include <skelly_sim.hpp>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include <fiber_chebyshev_constraint.hpp>
#include <kernels.hpp>
#include <periphery.hpp>
#include <utils.hpp>

#include <toml.hpp>

/// @file
/// @brief Implement FiberChebyshevConstraint class and associated functions

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

const std::string FiberChebyshevConstraint::BC_name[] = {"Force",           "Torque",   "Velocity",
                                                      "AngularVelocity", "Position", "Angle"};
