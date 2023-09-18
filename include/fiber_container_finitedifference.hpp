#ifndef FIBER_CONTAINER_FINITEDIFFERENCE_HPP
#define FIBER_CONTAINER_FINITEDIFFERENCE_HPP

#include <fiber_container_base.hpp>
#include <fiber_finitedifference.hpp>
#include <kernels.hpp>
#include <params.hpp>
#include <skelly_sim.hpp>

class Periphery;
class BodyContainer;

/// Finite difference fiber container class

class FiberContainerFinitedifference : public FiberContainerBase {
  public:
    //! \name Constructors and destructors
    //@{

    /// Constructor using TOML and params
    FiberContainerFinitedifference(toml::array &fiber_tables, Params &params);

    //@}

    //! \name Public member functions
    //@{

    /// Initialize the fiber container
    void init_fiber_container(toml::array &fiber_tables, Params &params) override;

    /// Update the local (MPI) node positions
    void update_local_node_positions() override;

    /// Begin of an iterator to this object
    ActiveIterator<FiberFiniteDifference> begin() { return ActiveIterator<FiberFiniteDifference>(0, fibers_); }

    /// End of an iterator to this object
    ActiveIterator<FiberFiniteDifference> end() {
        return ActiveIterator<FiberFiniteDifference>(fibers_.size(), fibers_);
    }

    /// Begin of a const iterator to this object
    ActiveIterator<FiberFiniteDifference> begin() const {
        return ActiveIterator<FiberFiniteDifference>(0, const_cast<decltype(fibers_) &>(fibers_));
    }

    /// End of a const iterator to this object
    ActiveIterator<FiberFiniteDifference> end() const {
        return ActiveIterator<FiberFiniteDifference>(fibers_.size(), const_cast<decltype(fibers_) &>(fibers_));
    }

    //@}

    //! \name Public member variables
    //@{

    std::vector<FiberFiniteDifference> fibers_; ///< Array of fibers local to this MPI rank

    //@}
};

#endif
