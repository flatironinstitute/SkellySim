#ifndef FIBER_HPP
#define FIBER_HPP
#include <Eigen/Dense>
#include <Eigen/LU>
#include <unordered_map>
#include <vector>

#include <kernels.hpp>
#include <params.hpp>

#include <toml.hpp>


/// @brief Class to represent a single flexible filament
///
/// Actions on the fiber class are typically handled via the container object, which will
/// distribute calls appropriately across all fibers in the container.
class Fiber {
  public:
    enum BC { Force, Torque, Velocity, AngularVelocity, Position, Angle };

    int num_points_;               ///< number of points representing the fiber
    double length_;                ///< length of fiber
    double bending_rigidity_;      ///< bending rigidity 'E' of fiber
    double penalty_param_ = 500.0; ///< @brief Tension penalty parameter for linear operator @see update_linear_operator
    /// @brief scale of external force on node @see generate_external_force
    /// \f[{\bf f} = f_s * {\bf x}_s\f]
    double force_scale_ = 4.0;
    // FIXME: Magic numbers in linear operator calculation
    double beta_tstep_ = 1.0; ///< penalty parameter to ensure inextensibility
    double epsilon_ = 1E-3;   ///< slenderness parameter
    double v_length_ = 0.0;   ///< fiber growth velocity

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

    Eigen::MatrixXd x_;     ///< [ 3 x num_points_ ] matrix representing coordinates of fiber points
    Eigen::MatrixXd xs_;    ///< [ 3 x num_points_ ] matrix representing first derivative of fiber points
    Eigen::MatrixXd xss_;   ///< [ 3 x num_points_ ] matrix representing second derivative of fiber points
    Eigen::MatrixXd xsss_;  ///< [ 3 x num_points_ ] matrix representing third derivative of fiber points
    Eigen::MatrixXd xssss_; ///< [ 3 x num_points_ ] matrix representing fourth derivative of fiber points

    /// [ 3*num_points_ x 3*num_points_] Oseen tensor for fiber @see Fiber::update_stokeslet
    Eigen::MatrixXd stokeslet_;

    Eigen::MatrixXd A_;                         ///< Fiber's linear operator for matrix solver
    Eigen::PartialPivLU<Eigen::MatrixXd> A_LU_; ///< Fiber preconditioner, LU decomposition of Fiber::A_
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

    /// Map of cached matrices for different values of num_points_. Calculated automagically at program start. @see
    /// compute_matrices
    const static std::unordered_map<int, fib_mat_t> matrices_;

    Fiber(toml::table *fiber_table, double eta);

    /// @brief initialize empty fiber
    /// @param[in] num_points fiber 'resolution'
    /// @param[in] bending_rigidity bending rigidity of fiber
    /// @param[in] eta fluid viscosity
    ///
    /// @deprecated Initializing with a toml::table structure is the preferred initialization. This is only around for
    /// testing.
    Fiber(int num_points, double bending_rigidity, double eta)
        : num_points_(num_points), bending_rigidity_(bending_rigidity) {
        init(eta);
    };

    ///< @brief Set some default values and resize arrays
    ///
    ///< _MUST_ be called from constructors.
    ///
    /// Initializes: Fiber::x_, Fiber::xs_, Fiber::xss_, Fiber::xsss_, Fiber::xssss_, Fiber::c_0_, Fiber::c_1_
    void init(double eta) {
        x_ = Eigen::MatrixXd::Zero(3, num_points_);
        x_.row(0) = Eigen::ArrayXd::LinSpaced(num_points_, 0, 1.0).transpose();
        xs_.resize(3, num_points_);
        xss_.resize(3, num_points_);
        xsss_.resize(3, num_points_);
        xssss_.resize(3, num_points_);

        c_0_ = -log(M_E * std::pow(epsilon_, 2)) / (8 * M_PI * eta);
        c_1_ = 2.0 / (8.0 * M_PI * eta);
    };

    void update_preconditioner();
    void update_force_operator();
    void update_RHS(double dt, const Eigen::Ref<const Eigen::MatrixXd> flow,
                    const Eigen::Ref<const Eigen::MatrixXd> f_external);
    void update_linear_operator(double dt, double eta);
    void apply_bc_rectangular(double dt, const Eigen::Ref<const Eigen::MatrixXd> &v_on_fiber,
                              const Eigen::Ref<const Eigen::MatrixXd> &f_on_fiber);
    void translate(const Eigen::Vector3d &r) { x_.colwise() += r; };
    void update_derivatives();
    void update_stokeslet(double);
};

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
    std::vector<Fiber> fibers; ///< Array of fibers local to this MPI rank
    /// pointer to FMM object (pointer to avoid constructing fmm_ with default FiberContainer)
    std::unique_ptr<kernels::FMM<stkfmm::Stk3DFMM>> fmm_;

    /// Empty container constructor to avoid initialization list complications. No way to
    /// initialize after using this constructor, so overwrite objects with full constructor.
    FiberContainer(){};
    FiberContainer(toml::array *fiber_tables, Params &params);

    void update_derivatives();
    void update_stokeslets(double eta);
    void update_linear_operators(double dt, double eta);

    /// @brief get total number of points across fibers in the container
    /// Usually you need this to form arrays used as input later
    /// @returns total number of points across fibers in the container :)
    int get_local_total_fib_points() const {
        // FIXME: This could certainly be cached
        int tot = 0;
        for (auto &fib : fibers)
            tot += fib.num_points_;
        return tot;
    };

    int get_global_total_fib_points() const;

    /// @brief Get the size of all local fibers contribution to the matrix problem solution
    int get_local_solution_size() const { return get_local_total_fib_points() * 4; }

    Eigen::MatrixXd generate_constant_force() const;
    Eigen::MatrixXd get_r_vectors() const;
    Eigen::VectorXd get_RHS() const;
    Eigen::MatrixXd flow(const Eigen::Ref<const Eigen::MatrixXd> &forces,
                         const Eigen::Ref<const Eigen::MatrixXd> &r_trg_external, double eta) const;
    Eigen::VectorXd matvec(const Eigen::Ref<const Eigen::VectorXd> &x_all,
                           const Eigen::Ref<const Eigen::MatrixXd> &v_fib) const;
    Eigen::MatrixXd apply_fiber_force(const Eigen::Ref<const Eigen::VectorXd> &x_all) const;
    Eigen::VectorXd apply_preconditioner(const Eigen::Ref<const Eigen::VectorXd> &x_all) const;
};

#endif
