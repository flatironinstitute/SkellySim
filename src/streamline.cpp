#include <streamline.hpp>
#include <system.hpp>

#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>

typedef Eigen::Vector3d point_type;

void get_velocity_at_point(const point_type &x, point_type &dxdt, const double /* t */) {
    Eigen::VectorXd v = System::velocity_at_targets(x);
    dxdt = {v[0], v[1], v[2]};
}

// TODO: store velocities as well!
struct push_back_points_and_time {
    std::vector<point_type> &m_points;
    std::vector<double> &m_times;

    push_back_points_and_time(std::vector<point_type> &points, std::vector<double> &times)
        : m_points(points), m_times(times) {}

    void operator()(const point_type &x, double t) {
        m_points.push_back(x);
        m_times.push_back(t);
    }
};

void StreamLine::compute() {
    using namespace boost::numeric::odeint;
    double a_x = 1.0, a_dxdt = 1.0;
    typedef runge_kutta_cash_karp54<point_type> error_stepper_type;
    typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;
    controlled_stepper_type controlled_stepper(
        default_error_checker<double, range_algebra, default_operations>(abs_err, rel_err, a_x, a_dxdt));

    point_type x0 = {x(0), x(1), x(2)};
    std::vector<point_type> xvec;
    std::vector<double> times;

    integrate_adaptive(controlled_stepper, get_velocity_at_point, x0, 0.0, t_final, dt_init,
                       push_back_points_and_time(xvec, times));

    x = MatrixMap((double *)xvec.data(), 3, xvec.size());
}
