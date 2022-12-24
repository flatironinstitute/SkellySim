#include <stdexcept>
#include <streamline.hpp>
#include <system.hpp>

#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>

#include <spdlog/spdlog.h>

typedef Eigen::Vector3d point_type;

void get_velocity_at_point(const point_type &x, point_type &dxdt, const double t) {
    Eigen::VectorXd v = System::velocity_at_targets(x);
    dxdt = point_type{v[0], v[1], v[2]};
}

point_type get_vorticity_at_point(const point_type &x) {
    double epsilon = 1E-7;
    Eigen::Matrix<double, 3, 6> x_plus_eps;
    x_plus_eps.col(0) = Eigen::Vector3d{x[0] + epsilon, x[1], x[2]};
    x_plus_eps.col(1) = Eigen::Vector3d{x[0] - epsilon, x[1], x[2]};
    x_plus_eps.col(2) = Eigen::Vector3d{x[0], x[1] + epsilon, x[2]};
    x_plus_eps.col(3) = Eigen::Vector3d{x[0], x[1] - epsilon, x[2]};
    x_plus_eps.col(4) = Eigen::Vector3d{x[0], x[1], x[2] + epsilon};
    x_plus_eps.col(5) = Eigen::Vector3d{x[0], x[1], x[2] - epsilon};

    Eigen::MatrixXd v = System::velocity_at_targets(x_plus_eps);

    return 0.5 *
           point_type{
               (v(2, 2) - v(2, 3)) - (v(1, 4) - v(1, 5)),
               (v(0, 4) - v(0, 5)) - (v(2, 0) - v(2, 1)),
               (v(1, 0) - v(1, 1)) - (v(0, 2) - v(0, 3)),
           } /
           epsilon;
}

void get_vorticity_at_point_integrand(const point_type &x, point_type &dxdt, const double t) {
    dxdt = get_vorticity_at_point(x);
}

struct push_back_points_and_time {
    std::vector<point_type> &m_points;
    std::vector<double> &m_times;

    push_back_points_and_time(std::vector<point_type> &points, std::vector<double> &times)
        : m_points(points), m_times(times) {}

    void operator()(const point_type &x, double t) {
        m_points.push_back(x);
        m_times.push_back(t);
        if (System::velocity_at_targets(x).norm() > 1E3)
            throw std::runtime_error("Possible singularity detected");
    }
};

Eigen::MatrixXd join_back_and_forward(MatrixRef &back, MatrixRef &forward) {
    Eigen::MatrixXd x(back.rows(), back.cols() + forward.cols() - 1);

    for (int i_col = 0; i_col < back.cols() - 1; ++i_col)
        x.col(i_col) = back.col(back.cols() - i_col - 1);

    x.block(0, back.cols() - 1, back.rows(), forward.cols()) = forward;

    return x;
}

void StreamLine::compute() {
    using namespace boost::numeric::odeint;
    double a_x = 1.0, a_dxdt = 1.0;
    typedef runge_kutta_cash_karp54<point_type> error_stepper_type;
    typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;
    controlled_stepper_type controlled_stepper(
        default_error_checker<double, range_algebra, default_operations>(abs_err, rel_err, a_x, a_dxdt));

    const point_type x0_orig(x.data());
    point_type x0 = x0_orig;
    std::vector<point_type> xvec;
    std::vector<double> times;
    std::vector<point_type> xvecneg;
    std::vector<double> timesneg;

    if (back_integrate) {
        try {
            integrate_adaptive(controlled_stepper, get_velocity_at_point, x0, 0.0, -t_final, -dt_init,
                               push_back_points_and_time(xvecneg, timesneg));
        } catch (std::runtime_error &e) {
            spdlog::warn("Streamline early exit: {}", e.what());
        }

        x0 = x0_orig;
    }
    try {
        integrate_adaptive(controlled_stepper, get_velocity_at_point, x0, 0.0, t_final, dt_init,
                           push_back_points_and_time(xvec, times));
    } catch (std::runtime_error &e) {
        spdlog::warn("Streamline early exit: {}", e.what());
    }

    MatrixMap xn((double *)xvecneg.data(), 3, xvecneg.size());
    MatrixMap xp((double *)xvec.data(), 3, xvec.size());
    VectorMap tn((double *)timesneg.data(), timesneg.size());
    VectorMap tp((double *)times.data(), times.size());

    if (back_integrate) {
        x = join_back_and_forward(xn, xp);
        time = join_back_and_forward(tn.transpose(), tp.transpose()).transpose();
    } else {
        x = xp;
        time = tp;
    }

    val = System::velocity_at_targets(x);
}

void VortexLine::compute() {
    using namespace boost::numeric::odeint;
    double a_x = 1.0, a_dxdt = 1.0;
    typedef runge_kutta_cash_karp54<point_type> error_stepper_type;
    typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;
    controlled_stepper_type controlled_stepper(
        default_error_checker<double, range_algebra, default_operations>(abs_err, rel_err, a_x, a_dxdt));

    const point_type x0_orig(x.data());
    point_type x0 = x0_orig;
    std::vector<point_type> xvec;
    std::vector<double> times;
    std::vector<point_type> xvecneg;
    std::vector<double> timesneg;

    if (back_integrate) {
        try {
            integrate_adaptive(controlled_stepper, get_vorticity_at_point_integrand, x0, 0.0, -t_final, -dt_init,
                               push_back_points_and_time(xvecneg, timesneg));
        } catch (std::runtime_error &e) {
            spdlog::warn("Streamline early exit: {}", e.what());
        }

        x0 = x0_orig;
    }

    try {
        integrate_adaptive(controlled_stepper, get_vorticity_at_point_integrand, x0, 0.0, t_final, dt_init,
                           push_back_points_and_time(xvec, times));
    } catch (std::runtime_error &e) {
        spdlog::warn("Streamline early exit: {}", e.what());
    }

    MatrixMap xp((double *)xvec.data(), 3, xvec.size());
    MatrixMap xn((double *)xvecneg.data(), 3, xvecneg.size());
    VectorMap tn((double *)timesneg.data(), timesneg.size());
    VectorMap tp((double *)times.data(), times.size());

    if (back_integrate) {
        x = join_back_and_forward(xn, xp);
        time = join_back_and_forward(tn.transpose(), tp.transpose()).transpose();
    } else {
        x = xp;
        time = tp;
    }

    val.resize(x.rows(), x.cols());

    for (int i = 0; i < val.cols(); ++i)
        val.col(i) = get_vorticity_at_point(x.col(i));
}
