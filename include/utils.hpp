#ifndef UTILS_HPP
#define UTILS_HPP

#include <skelly_sim.hpp>

namespace cnpy {
struct NpyArray;
using npz_t = std::map<std::string, NpyArray>;
} // namespace cnpy

namespace utils {

class CubicHermiteSplines {
  public:
    CubicHermiteSplines(const MatrixRef &x);
    CubicHermiteSplines(const MatrixRef &x, const VectorRef &tan_start, const VectorRef &tan_end);

    struct Panel {
        Eigen::Vector3d eval(double t) {
            const double t2 = t * t;
            const double t3 = t * t2;
            return c * Eigen::Vector4d{t3, t2, t, 1.0};
        };
        Eigen::Vector3d evald(double t) {
            const double t2 = t * t;
            return c * Eigen::Vector4d{3.0 * t2, 2.0 * t, 1.0, 0.0};
        };
        Eigen::Matrix<double, 3, 4> c;
    };

    double arc_length(const int n_nodes);
    std::vector<Panel> panels;
    Eigen::VectorXd dL;
    double L_tot;
};

Eigen::MatrixXd finite_diff(ArrayRef &s, int M, int n_s);
Eigen::VectorXd collect_into_global(VectorRef &local_vec);

Eigen::MatrixXd load_mat(cnpy::npz_t &npz, const char *var);
Eigen::VectorXd load_vec(cnpy::npz_t &npz, const char *var);

template <typename DerivedA, typename DerivedB>
bool allclose(
    const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b,
    const typename DerivedA::RealScalar &rtol = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
    const typename DerivedA::RealScalar &atol = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon()) {
    return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
}

class LoggerRedirect {
  public:
    LoggerRedirect(std::ostream &in) : m_orig(in), m_old_buffer(in.rdbuf(ss.rdbuf())) {}

    ~LoggerRedirect() { m_orig.rdbuf(m_old_buffer); }

    void flush(spdlog::level::level_enum level, std::string logger = "status") {
        std::istringstream dumbtmp(ss.str());
        for (std::string line; std::getline(dumbtmp, line);)
            spdlog::get(logger)->log(level, line);
        ss.str("");
        ss.clear();
    }

  private:
    LoggerRedirect(const LoggerRedirect &);
    LoggerRedirect &operator=(const LoggerRedirect &);

    std::stringstream ss;         ///< temporary object to redirect to for lifetime of this object
    std::ostream &m_orig;         ///< Actual original ostream object (usually cout)
    std::streambuf *m_old_buffer; ///< pointer to original streambuf pointer (usually cout's)
};

}; // namespace utils

#endif
