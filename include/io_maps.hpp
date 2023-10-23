#ifndef IO_MAPS_HPP
#define IO_MAPS_HPP

#include <body_container.hpp>
#include <fiber_container_base.hpp>
#include <fiber_container_finite_difference.hpp>
#include <fiber_finite_difference.hpp>
#include <periphery.hpp>
#include <serialization.hpp>

#include <msgpack.hpp>

/// @brief Structure for importing frame of trajectory into the simulation
///
/// We can't use output_map_t here, but rather a similar struct which uses copies of the member
/// variables (rather than references) which are then used to update the System variables.
typedef struct input_map_t {
    double time;                                                ///< System::properties
    double dt;                                                  ///< System::properties
    std::unique_ptr<FiberContainerBase> fibers;                 ///< System::fc_
    BodyContainer bodies;                                       ///< System::bc_
    Periphery shell;                                            ///< System::shell_
    std::vector<std::pair<std::string, std::string>> rng_state; ///< string representation of split/unsplit state in RNG
    MSGPACK_DEFINE_MAP(time, dt, rng_state, fibers, bodies, shell); ///< Helper routine to specify serialization
} input_map_t;

/// @brief Structure for trajectory output via msgpack
///
/// This can be extended easily, so long as you update the corresponding input_map_t and potentially the System::write()
/// function if you can't use a reference in the output_map for some reason.
typedef struct output_map_t {
    double &time;                                               ///< System::properties
    double &dt;                                                 ///< System::properties
    std::unique_ptr<FiberContainerBase> &fibers;                ///< System::fc_
    BodyContainer &bodies;                                      ///< System::bc_
    Periphery &shell;                                           ///< System::shell_
    std::vector<std::pair<std::string, std::string>> rng_state; ///< string representation of split/unsplit state in RNG
    MSGPACK_DEFINE_MAP(time, dt, rng_state, fibers, bodies, shell); ///< Helper routine to specify serialization
} output_map_t;

/// @brief Structure for trajectory header information via msgpack
///
/// Contains all of the header information for the simulation
typedef struct header_map_t {
    // Make sure that the trajectory version always comes first!!!!!
    int trajversion;               ///< SKELLYSIM_TRAJECTORY_VERSION
    int number_mpi_ranks;          ///< System::rank_
    int fiber_type;                ///< System::fc_->fiber_type_
    std::string skellysim_version; ///< SKELLYSIM_VERSION
    std::string skellysim_commit;  ///< SKELLYSIM_COMMIT
    std::string simdate;           ///< Date of simulation from chrono and ctime
    std::string hostname;          ///< Hostnames of the simulation
    MSGPACK_DEFINE_MAP(trajversion, number_mpi_ranks, fiber_type, skellysim_version, skellysim_commit, simdate,
                       hostname);
} header_map_t;

#endif
