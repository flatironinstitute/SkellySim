#include <system.hpp>
#include <fstream>

System::System(std::string &input_file) {
    toml::table config = toml::parse_file(input_file);
    params_ = Params(config.get_as<toml::table>("params"));
    fc_ = FiberContainer(config.get_as<toml::array>("fibers"), params_);
    shell_ = params_.body_precompute_file.length() ? Periphery(params_.body_precompute_file) : Periphery();
}
