#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <body.hpp>
#include <fiber.hpp>
#include <params.hpp>
#include <periphery.hpp>

class System {
public:
    Params params_;
    FiberContainer fc_;
    BodyContainer bc_;
    Periphery shell_;

    System(std::string &input_file);
};

#endif
