#ifndef FIBER_HPP
#define FIBER_HPP

#include <skelly_sim.hpp>

#include <Eigen/LU>
#include <list>
#include <unordered_map>

#include <kernels.hpp>
#include <params.hpp>

class Periphery;
class BodyContainer;

/// @brief Class to represent a single flexible filament
///
/// Actions on the fiber class are typically handled via the container object, which will
/// distribute calls appropriately across all fibers in the container.
class Fiber {
  public:
    enum BC { Force, Torque, Velocity, AngularVelocity, Position, Angle };
    static const std::string BC_name[];

    // Input parameters
    int n_nodes_;                  ///< number of nodes representing the fiber
    double radius_;                ///< radius of the fiber (for slender-body-theory, though possibly for collisions eventually)
    double length_;                ///< Desired 'constraint' length of fiber
    double length_prev_;           ///< Last accepted length_
    double bending_rigidity_;      ///< bending rigidity 'E' of fiber
    double penalty_param_ = 500.0; ///< @brief Tension penalty parameter for linear operator @see update_linear_operator
    /// @brief scale of external force on node @see generate_external_force
    /// \f[{\bf f} = f_s * {\bf x}_s\f]
    double force_scale_ = 0.0;
    // FIXME: Magic numbers in linear operator calculation
    double beta_tstep_ = 1.0; ///< penalty parameter to ensure inextensibility
    double epsilon_ = 1E-3;   ///< slenderness parameter
    bool minus_clamped_ = false;

    /// (body, site) pair for minus end binding. -1 implies unbound
    std::pair<int, int> binding_site_{-1, -1};

    double v_growth_ = 0.0;      ///< instantaneous fiber growth velocity
    bool near_periphery = false; ///< flag if interacting with periphery

    /// @brief Coefficient for SBT @see Fiber::init
    /// \f[ c_0 = -\frac{log(e \epsilon^\ell)}{8 \pi \eta}\f]
    double c_0_;

    /// @brief Coefficient for SBT @see Fiber::init
    /// \f[ c_1 = \frac{1}{4\pi\eta} \f]
    double c_1_;

    /// Boundary condition pair for minus end of fiber
    std::pair<BC, BC> bc_minus_ = {BC::Velocity, BC::AngularVelocity};
    /// Boundary condition pair for plus end of fiber
    std::pair<BC, BC> bc_plus_ = {BC::Force, BC::Torque};

    Eigen::VectorXd tension_; ///< [ n_nodes ] vector representing local tension on fiber nodes
    Eigen::MatrixXd x_;     ///< [ 3 x n_nodes_ ] matrix representing coordinates of fiber nodes
    Eigen::MatrixXd xs_;    ///< [ 3 x n_nodes_ ] matrix representing first derivative of fiber nodes
    Eigen::MatrixXd xss_;   ///< [ 3 x n_nodes_ ] matrix representing second derivative of fiber nodes
    Eigen::MatrixXd xsss_;  ///< [ 3 x n_nodes_ ] matrix representing third derivative of fiber nodes
    Eigen::MatrixXd xssss_; ///< [ 3 x n_nodes_ ] matrix representing fourth derivative of fiber nodes

    /// [ 3*n_nodes_ x 3*n_nodes_] Oseen tensor for fiber @see Fiber::update_stokeslet
    Eigen::MatrixXd stokeslet_;

    Eigen::MatrixXd A_;                      ///< Fiber's linear operator for matrix solver
    Eigen::FullPivLU<Eigen::MatrixXd> A_LU_; ///< Fiber preconditioner, LU decomposition of Fiber::A_
    /// Fiber force operator, @see Fiber::update_force_operator, FiberContainer::apply_fiber_force
    Eigen::MatrixXd force_operator_;
    Eigen::VectorXd RHS_; ///< Current 'right-hand-side' for matrix formulation of solver

    /// Structure that caches arrays useful for calculating various fiber values
    typedef struct {
        Eigen::ArrayXd alpha;
        Eigen::ArrayXd alpha_roots;
        Eigen::ArrayXd alpha_tension;
        Eigen::ArrayXd weights_0;
        Eigen::MatrixXd D_1_0;
        Eigen::MatrixXd D_2_0;
        Eigen::MatrixXd D_3_0;
        Eigen::MatrixXd D_4_0;
        Eigen::MatrixXd P_X;
        Eigen::MatrixXd P_T;
        Eigen::MatrixXd P_downsample_bc;
    } fib_mat_t;

    /// Map of cached matrices for different values of n_nodes_. Calculated automagically at program start. @see
    /// compute_matrices
    const static std::unordered_map<int, fib_mat_t> matrices_;

    Fiber(toml::value &fiber_table, double eta);
    Fiber() = default;

    /// @brief initialize empty fiber
    /// @param[in] n_nodes fiber 'resolution'
    /// @param[in] bending_rigidity bending rigidity of fiber
    /// @param[in] eta fluid viscosity
    ///
    /// @deprecated Initializing with a toml::table structure is the preferred initialization. This is only around for
    /// testing.
    Fiber(int n_nodes, double bending_rigidity, double eta) : n_nodes_(n_nodes), bending_rigidity_(bending_rigidity) {
        init(eta);
        length_prev_ = length_;
    };

    Fiber(const Fiber &old_fib, const double eta) {
        *this = old_fib;
        init(eta);
    };

    ///< @brief Set some default values and resize arrays
    ///
    ///< _MUST_ be called from constructors.
    ///
    /// Initializes: Fiber::x_, Fiber::xs_, Fiber::xss_, Fiber::xsss_, Fiber::xssss_, Fiber::c_0_, Fiber::c_1_
    void init(double eta) {
        if (x_.size() != 3 * n_nodes_) {
            x_ = Eigen::MatrixXd::Zero(3, n_nodes_);
            x_.row(0) = Eigen::ArrayXd::LinSpaced(n_nodes_, 0, 1.0).transpose();
        }
        xs_.resize(3, n_nodes_);
        xss_.resize(3, n_nodes_);
        xsss_.resize(3, n_nodes_);
        xssss_.resize(3, n_nodes_);
    };

    void update_constants(double eta) {
        epsilon_ = radius_ / length_;
        c_0_ = -log(M_E * std::pow(epsilon_, 2)) / (8 * M_PI * eta);
        c_1_ = 2.0 / (8.0 * M_PI * eta);
    }
    void update_preconditioner();
    void update_force_operator();
    void update_RHS(double dt, MatrixRef &flow, MatrixRef &f_external);
    void update_linear_operator(double dt, double eta);
    void apply_bc_rectangular(double dt, MatrixRef &v_on_fiber, MatrixRef &f_on_fiber);
    void translate(const Eigen::Vector3d &r) { x_.colwise() += r; };
    void update_derivatives();
    void update_stokeslet(double);
    bool attached_to_body() { return binding_site_.first >= 0; };
    bool is_minus_clamped() { return minus_clamped_ || attached_to_body(); };
#ifndef SKELLY_DEBUG
    MSGPACK_DEFINE_MAP(n_nodes_, length_, length_prev_, bending_rigidity_, penalty_param_, force_scale_, beta_tstep_,
                       epsilon_, binding_site_, tension_, x_);
#else
    MSGPACK_DEFINE_MAP(n_nodes_, length_, length_prev_, bending_rigidity_, penalty_param_, force_scale_, beta_tstep_,
                       epsilon_, binding_site_, tension_, x_, A_, RHS_, force_operator_, bc_minus_, bc_plus_, xs_, xss_, xsss_,
                       xssss_, stokeslet_);
#endif
};

MSGPACK_ADD_ENUM(Fiber::BC);

/// Class to hold the fiber objects.
///
/// The container object is designed to work on fibers local to that MPI rank. Each MPI rank
/// should have its own container, with its own unique fibers. The container object does not
/// have any knowledge of the MPI world state, which, for example, is passed in externally to
/// the FiberContainer::flow method and potentially others.
///
/// Developer note: ideally all interactions with the fiber objects should be through this
/// container, except for testing purposes. Operating on fibers outside of the container class
/// is ill-advised.
class FiberContainer {
  public:
    std::list<Fiber> fibers; ///< Array of fibers local to this MPI rank
    /// pointer to FMM object (pointer to avoid constructing stokeslet_kernel_ with default FiberContainer)
    std::shared_ptr<kernels::FMM<stkfmm::Stk3DFMM>> stokeslet_kernel_;

    /// Empty container constructor to avoid initialization list complications. No way to
    /// initialize after using this constructor, so overwrite objects with full constructor.
    FiberContainer() = default;
    FiberContainer(toml::array &fiber_tables, Params &params);

    void update_derivatives();
    void update_stokeslets(double eta);
    void update_linear_operators(double dt, double eta);
    void update_cache_variables(double dt, double eta);
    void update_RHS(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers);
    void apply_bc_rectangular(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers);
    void step(VectorRef &fiber_sol);
    void repin_to_bodies(BodyContainer &bodies);

    /// @brief get total number of nodes across fibers in the container
    /// Usually you need this to form arrays used as input later
    /// @returns total number of nodes across fibers in the container :)
    int get_local_node_count() const {
        // FIXME: This could certainly be cached
        int tot = 0;
        for (auto &fib : fibers)
            tot += fib.n_nodes_;
        return tot;
    };

    int get_global_total_fib_nodes() const;

    /// @brief Get the size of all local fibers contribution to the matrix problem solution
    int get_local_solution_size() const { return get_local_node_count() * 4; }

    /// @brief Get number of local fibers
    int get_local_count() const { return fibers.size(); };

    int get_global_count() const;

    Eigen::MatrixXd generate_constant_force() const;
    Eigen::MatrixXd get_local_node_positions() const;
    Eigen::VectorXd get_RHS() const;
    Eigen::MatrixXd flow(const MatrixRef &r_trg, const MatrixRef &forces, double eta, bool subtract_self = true) const;
    Eigen::VectorXd matvec(VectorRef &x_all, MatrixRef &v_fib, MatrixRef &v_fib_boundary) const;
    Eigen::MatrixXd apply_fiber_force(VectorRef &x_all) const;
    Eigen::VectorXd apply_preconditioner(VectorRef &x_all) const;

    void update_boundary_conditions(Periphery &shell, bool periphery_binding_flag);

  private:
    int world_size_ = -1;
    int world_rank_;

  public:
    MSGPACK_DEFINE(fibers);
};

#endif
