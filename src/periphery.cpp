#include <periphery.hpp>

SphericalPeriphery::SphericalPeriphery(int n_nodes, double radius) {
    const double phi = (1 + sqrt(5.0)) / 2; // Golden radio. Neat!
    const int N = n_nodes / 2;
    constexpr double pi = M_PI;
    if (n_nodes % 2 != 0) {
        std::cerr << "SphericalPeriphery only supports even numbers of nodes\n";
        exit(1);
    }

    quadrature_nodes_ = Eigen::MatrixXd::Zero(3, n_nodes);
    nodes_normal_ = Eigen::MatrixXd::Zero(3, n_nodes);

    for (int i = -N; i < N; ++i) {
        double lat = std::asin((2.0 * i) / (2 * N + 1));
        double lon = (std::fmod(i, phi)) * 2 * pi / phi;
        if (lon < -pi)
            lon += 2 * pi;
        else if (lon > pi)
            lon -= 2 * pi;
        quadrature_nodes_.col(i + N) = Eigen::Vector3d({cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)});
    }
    quadrature_nodes_ *= radius;

    nodes_normal_ = gradh(quadrature_nodes_).colwise().normalized();
}

Eigen::VectorXd SphericalPeriphery::h(const Eigen::Ref<Eigen::MatrixXd> &p) {
    return p.array().pow(2).colwise().sum() - std::pow(radius_, 2);
}

Eigen::MatrixXd SphericalPeriphery::gradh(const Eigen::Ref<Eigen::MatrixXd> &p) { return 2 * p; }
