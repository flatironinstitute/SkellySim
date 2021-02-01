#include <system.hpp>
#include <fstream>

System::System(std::string &input_file) {
    toml::table config = toml::parse_file(input_file);
    params_ = Params(config.get_as<toml::table>("params"));
    fc_ = FiberContainer(config.get_as<toml::array>("fibers"), params_);
    shell_ = params_.shell_precompute_file.length() ? Periphery(params_.shell_precompute_file) : Periphery();
    bc_ = BodyContainer(config.get_as<toml::array>("bodies"), params_);
}
