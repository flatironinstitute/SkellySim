#ifndef LINK_HPP
#define LINK_HPP

#include <skelly_sim.hpp>

#include <vector>

class FiberContainer;

struct Link {
    Vector3d pos;
};

class LinkContainer {
  protected:
    using sublist = std::vector<unsigned int>;
    void swap_state(const std::size_t &index, sublist &from, sublist &to) {
        std::size_t link_index = from[index];
        from[index] = from.back();
        from.pop_back();
        to.push_back(link_index);
    }

  public:
    double capture_radius_ = 0.5;
    double k_on_ = 0.1;
    double k_off_ = 0.1;

    void insert(const toml::value &link_config);

    LinkContainer() = default;
    LinkContainer(toml::value &link_group_table);

    void queue_for_attachment(const std::pair<std::size_t, global_fiber_pointer> &pair) {
        attachment_queue_.push_back(pair);
    }
    void sync_attachments(FiberContainer &fc);
    void activate(const std::size_t &inactive_index);

    void deactivate(const std::size_t &active_index);

    const int n_inactive() const { return inactive_.size(); }
    const int n_active() const { return active_.size(); }
    const int size() const { return links.size(); }

    const sublist &inactive() const { return inactive_; }
    const sublist &active() const { return active_; }

    const Link operator[](std::size_t index) const { return links[index]; }

    void kmc_step(const double &dt);

  protected:
    int mpi_rank_;
    int mpi_size_ = -1;

    std::vector<Link> links;
    sublist inactive_;
    sublist active_;

    // FIXME: THIS HAS TO BE REBUILT ON RESUME. RELIES ON RUNTIME VALUES (pointers yippee)!!
    std::vector<global_fiber_pointer> attached_;
    sublist detached_;
    std::vector<std::pair<std::size_t, global_fiber_pointer>> attachment_queue_;

    void attach(const std::size_t &link_id, const global_fiber_pointer &p, FiberContainer &fc);
    void detach(const std::size_t &link_id, FiberContainer &fc);
};

class BodyFiberLinkContainer : public LinkContainer {
  public:
    BodyFiberLinkContainer() = default;
    BodyFiberLinkContainer(toml::value &link_group_table) : LinkContainer(link_group_table){};
};

class PointFiberLinkContainer : public LinkContainer {
  public:
    PointFiberLinkContainer() = default;
    PointFiberLinkContainer(toml::value &link_group_table) : LinkContainer(link_group_table){};
};

#endif
