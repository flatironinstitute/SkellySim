#ifndef IO_MAPS_HPP
#define IO_MAPS_HPP

#include <fiber.hpp>
#include <periphery.hpp>

#include <msgpack.hpp>

/// @brief Structure for importing frame of trajectory into the simulation
///
/// We can't use output_map_t here, but rather a similar struct which uses copies of the member
/// variables (rather than references) which are then used to update the System variables.
typedef struct input_map_t {
    double time;                                                ///< System::properties
    double dt;                                                  ///< System::properties
    FiberContainer fibers;                                      ///< System::fc_
    Periphery shell;                                            ///< System::bc_
    std::vector<std::pair<std::string, std::string>> rng_state; ///< string representation of split/unsplit state in RNG
    MSGPACK_DEFINE_MAP(time, dt, rng_state, fibers, shell);     ///< Helper routine to specify serialization
} input_map_t;

/// @brief Structure for trajectory output via msgpack
///
/// This can be extended easily, so long as you update the corresponding input_map_t and potentially the System::write()
/// function if you can't use a reference in the output_map for some reason.
typedef struct output_map_t {
    double &time;                                               ///< System::properties
    double &dt;                                                 ///< System::properties
    FiberContainer &fibers;                                     ///< System::fc_
    Periphery &shell;                                           ///< System::shell_
    std::vector<std::pair<std::string, std::string>> rng_state; ///< string representation of split/unsplit state in RNG
    MSGPACK_DEFINE_MAP(time, dt, rng_state, fibers, shell);     ///< Helper routine to specify serialization
} output_map_t;

#endif
