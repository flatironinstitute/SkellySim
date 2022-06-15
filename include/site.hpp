#ifndef SITE_HPP
#define SITE_HPP

#include <skelly_sim.hpp>

#include <vector>

class SiteContainer {
  private:
    using sublist = std::vector<unsigned int>;
    void swap_state(const std::size_t &index, sublist &from, sublist &to) {
        std::size_t site_index = from[index];
        from[index] = from.back();
        from.pop_back();
        to.push_back(site_index);
    }

  public:
    double capture_radius_ = 0.5; // FIXME: need different constructor to specify SC types

    void insert(const toml::value &site_config);

    SiteContainer() = default;
    SiteContainer(toml::array &site_tables);

    void bind(const std::size_t &active_index) { swap_state(active_index, active_, bound_); }
    void unbind(const std::size_t &bound_index) { swap_state(bound_index, bound_, active_); }
    void activate(const std::size_t &inactive_index) { swap_state(inactive_index, inactive_, active_); }
    void deactivate(const std::size_t &active_index) { swap_state(active_index, active_, inactive_); }

    const int n_inactive() const { return inactive_.size(); }
    const int n_active() const { return active_.size(); }
    const int n_bound() const { return bound_.size(); }
    const int size() const { return pos_.cols(); }

    const sublist &inactive() const { return inactive_; }
    const sublist &active() const { return active_; }
    const sublist &bound() const { return bound_; }

    const Eigen::Vector3d operator[](std::size_t index) const { return pos_.col(index); }

  private:
    Eigen::Matrix3Xd pos_;
    sublist inactive_;
    sublist active_;
    sublist bound_;
};

#endif
