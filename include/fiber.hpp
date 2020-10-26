#ifndef FIBER_HPP
#define FIBER_HPP
#include <Eigen/Dense>
#include <Eigen/LU>
#include <unordered_map>
#include <vector>

class Fiber {
  public:
    enum BC { Force, Torque, Velocity, AngularVelocity, Position, Angle };

    int num_points_;
    double length_;
    double bending_rigidity_;
    double penalty_param_ = 500.0;
    // FIXME: Magic numbers in linear operator calculation
    double beta_tstep_ = 1.0;
    double epsilon_ = 1E-3;
    double v_length_ = 0.0;
    double c_0_, c_1_;

    std::pair<BC, BC> bc_minus_ = {BC::Velocity, BC::AngularVelocity};
    std::pair<BC, BC> bc_plus_ = {BC::Force, BC::Torque};

    typedef Eigen::MatrixXd matrix_t;
    typedef Eigen::ArrayXd array_t;
    matrix_t x_;
    matrix_t xs_;
    matrix_t xss_;
    matrix_t xsss_;
    matrix_t xssss_;
    matrix_t stokeslet_;

    matrix_t A_;
    Eigen::PartialPivLU<matrix_t> A_LU_;
    matrix_t force_operator_;
    Eigen::VectorXd RHS_;

    typedef struct {
        array_t alpha;
        array_t alpha_roots;
        array_t alpha_tension;
        array_t weights_0;
        matrix_t D_1_0;
        matrix_t D_2_0;
        matrix_t D_3_0;
        matrix_t D_4_0;
        matrix_t P_X;
        matrix_t P_T;
        matrix_t P_downsample_bc;
    } fib_mat_t;
    const static std::unordered_map<int, fib_mat_t> matrices_;

    Fiber(int num_points, double bending_rigidity, double eta)
        : num_points_(num_points), bending_rigidity_(bending_rigidity) {
        x_ = Eigen::MatrixXd::Zero(3, num_points);
        x_.row(0) = Eigen::VectorXd::LinSpaced(num_points, 0, 1.0).transpose();
        xs_.resize(3, num_points);
        xss_.resize(3, num_points);
        xsss_.resize(3, num_points);
        xssss_.resize(3, num_points);

        c_0_ = -log(M_E * std::pow(epsilon_, 2)) / (8 * M_PI * eta);
        c_1_ = 2.0 / (8.0 * M_PI * eta);
    };

    void build_preconditioner();
    void form_force_operator();
    void compute_RHS(double dt, const Eigen::Ref<Eigen::MatrixXd> flow, const Eigen::Ref<Eigen::MatrixXd> f_external);
    void form_linear_operator(double dt, double eta);
    void apply_bc_rectangular(double dt, const Eigen::Ref<Eigen::MatrixXd> &v_on_fiber,
                              const Eigen::Ref<Eigen::MatrixXd> &f_on_fiber);
    void translate(const Eigen::Vector3d &r) { x_.colwise() += r; };
    void update_derivatives();
    void update_stokeslet(double);
};

class FiberContainer {
  public:
    std::vector<Fiber> fibers;
    double slenderness_ratio;

    FiberContainer(int num_fibers, int num_points, double bending_rigidity, double eta) {
        fibers.reserve(num_fibers);
        for (int i = 0; i < num_fibers; ++i) {
            fibers.push_back(Fiber(num_points, bending_rigidity, eta));
        }
    }

    void update_derivatives();
    void update_stokeslets(double eta);
    void form_linear_operators(double dt, double eta);
    Eigen::MatrixXd flow(const Eigen::Ref<Eigen::MatrixXd> &forces, double eta) const;
    Eigen::VectorXd matvec(const Eigen::Ref<Eigen::VectorXd> &x_all, const Eigen::Ref<Eigen::MatrixXd> &v_fib) const;
    Eigen::MatrixXd apply_fiber_force(const Eigen::Ref<Eigen::VectorXd> &x_all) const;
    Eigen::VectorXd apply_preconditioner(const Eigen::Ref<Eigen::VectorXd> &x_all) const;
};

#endif
