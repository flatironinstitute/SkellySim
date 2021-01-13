#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <params.hpp>
#include <fiber.hpp>
#include <periphery.hpp>

class System {
public:
    Params params_;
    FiberContainer fc_;
    Periphery shell_;

    System(std::string &input_file);
};

#endif
