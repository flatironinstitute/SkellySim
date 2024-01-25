#ifndef FIBER_FINITE_DIFFERENCE_HPP
#define FIBER_FINITE_DIFFERENCE_HPP

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
class FiberFiniteDifference {
  public:
    enum BC { Force, Torque, Velocity, AngularVelocity, Position, Angle };
    static const std::string BC_name[];

    // Input parameters
    int n_nodes_;        ///< number of nodes representing the fiber
    double radius_;      ///< radius of the fiber (for slender-body-theory, though possibly for collisions eventually)
    double length_;      ///< Desired 'constraint' length of fiber
    double length_prev_; ///< Last accepted length_
    double bending_rigidity_;      ///< bending rigidity 'E' of fiber
    double penalty_param_ = 500.0; ///< @brief Tension penalty parameter for linear operator @see update_linear_operator
    /// @brief scale of external force on node @see generate_external_force
    /// \f[{\bf f} = f_s * {\bf x}_s\f]
    double force_scale_ = 0.0;
    // FIXME: Magic numbers in linear operator calculation
    double beta_tstep_ = 1.0;    ///< penalty parameter to ensure inextensibility
    double epsilon_;             ///< slenderness parameter
    bool minus_clamped_ = false; ///< Fix minus end in space with clamped condition

    /// (body, site) pair for minus end binding. -1 implies unbound
    std::pair<int, int> binding_site_{-1, -1};

    double v_growth_ = 0.0; ///< instantaneous fiber growth velocity

    /// @brief Coefficient for SBT @see FiberFiniteDifference::init
    /// \f[ c_0 = -\frac{log(e \epsilon^\ell)}{8 \pi \eta}\f]
    double c_0_;

    /// @brief Coefficient for SBT @see FiberFiniteDifference::init
    /// \f[ c_1 = \frac{1}{4\pi\eta} \f]
    double c_1_;

    /// Boundary condition pair for minus end of fiber
    std::pair<BC, BC> bc_minus_ = {BC::Velocity, BC::AngularVelocity};
    /// Boundary condition pair for plus end of fiber
    std::pair<BC, BC> bc_plus_ = {BC::Force, BC::Torque};

    Eigen::VectorXd tension_; ///< [ n_nodes ] vector representing local tension on fiber nodes
    Eigen::MatrixXd x_;       ///< [ 3 x n_nodes_ ] matrix representing coordinates of fiber nodes
    Eigen::MatrixXd xs_;      ///< [ 3 x n_nodes_ ] matrix representing first derivative of fiber nodes
    Eigen::MatrixXd xss_;     ///< [ 3 x n_nodes_ ] matrix representing second derivative of fiber nodes
    Eigen::MatrixXd xsss_;    ///< [ 3 x n_nodes_ ] matrix representing third derivative of fiber nodes
    Eigen::MatrixXd xssss_;   ///< [ 3 x n_nodes_ ] matrix representing fourth derivative of fiber nodes

    /// [ 3*n_nodes_ x 3*n_nodes_] Oseen tensor for fiber @see FiberFiniteDifference::update_stokeslet
    Eigen::MatrixXd stokeslet_;

    Eigen::MatrixXd A_; ///< FiberFiniteDifference's linear operator for matrix solver
    Eigen::FullPivLU<Eigen::MatrixXd>
        A_LU_; ///< FiberFiniteDifference preconditioner, LU decomposition of FiberFiniteDifference::A_
    /// FiberFiniteDifference force operator, @see FiberFiniteDifference::update_force_operator,
    /// FiberFiniteDifferenceContainer::apply_fiber_force
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

    FiberFiniteDifference(toml::value &fiber_table, double eta);
    FiberFiniteDifference() = default;

    /// @brief initialize empty fiber
    /// @param[in] n_nodes fiber 'resolution'
    /// @param[in] radius fiber radius
    /// @param[in] length fiber length
    /// @param[in] bending_rigidity bending rigidity of fiber
    /// @param[in] eta fluid viscosity
    ///
    /// @deprecated Initializing with a toml::table structure is the preferred initialization. This is only around for
    /// testing.
    FiberFiniteDifference(int n_nodes, double radius, double length, double bending_rigidity, double eta)
        : n_nodes_(n_nodes), radius_(radius), length_(length), bending_rigidity_(bending_rigidity) {
        init();
        length_prev_ = length_;
        update_constants(eta);
    };

    FiberFiniteDifference(const FiberFiniteDifference &old_fib, const double eta) {
        *this = old_fib;
        update_constants(eta);
    };

    bool active() const { return true; }

    ///< @brief Set some default values and resize arrays
    ///
    ///< _MUST_ be called from constructors.
    ///
    /// Initializes: FiberFiniteDifference::x_, FiberFiniteDifference::xs_, FiberFiniteDifference::xss_,
    /// FiberFiniteDifference::xsss_, FiberFiniteDifference::xssss_, FiberFiniteDifference::c_0_,
    /// FiberFiniteDifference::c_1_
    void init() {
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

    Eigen::VectorXd matvec(CVectorRef x, CMatrixRef v, CVectorRef v_boundary) const;
    void update_preconditioner();
    void update_force_operator();
    void update_boundary_conditions(Periphery &shell, const periphery_binding_t &periphery_binding);
    void update_RHS(double dt, CMatrixRef &flow, CMatrixRef &f_external);
    void update_linear_operator(double dt, double eta);
    void apply_bc_rectangular(double dt, CMatrixRef &v_on_fiber, CMatrixRef &f_on_fiber);
    void translate(const Eigen::Vector3d &r) { x_.colwise() += r; };
    void update_derivatives();
    void update_stokeslet(double);
    bool attached_to_body() const { return binding_site_.first >= 0; };
    bool is_minus_clamped() const { return minus_clamped_ || attached_to_body(); };
    bool is_plus_pinned() { return bc_plus_.first == BC::Velocity; };
#ifndef SKELLY_DEBUG
    MSGPACK_DEFINE_MAP(n_nodes_, radius_, length_, length_prev_, bending_rigidity_, penalty_param_, force_scale_,
                       beta_tstep_, binding_site_, tension_, x_, minus_clamped_);
#else
    MSGPACK_DEFINE_MAP(n_nodes_, radius_, length_, length_prev_, bending_rigidity_, penalty_param_, force_scale_,
                       beta_tstep_, binding_site_, tension_, x_, minus_clamped_, A_, RHS_, force_operator_, bc_minus_,
                       bc_plus_, xs_, xss_, xsss_, xssss_, stokeslet_);
#endif
};

MSGPACK_ADD_ENUM(FiberFiniteDifference::BC);

#endif
