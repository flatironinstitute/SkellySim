#ifndef UTILS_HPP
#define UTILS_HPP

#include <skelly_sim.hpp>
#include <spdlog/spdlog.h>

namespace utils {
Eigen::MatrixXd finite_diff(ArrayRef &s, int M, int n_s);
Eigen::VectorXd collect_into_global(VectorRef &local_vec);

class LoggerRedirect {
  public:
    LoggerRedirect(std::ostream &in) : m_orig(in), m_old_buffer(in.rdbuf(ss.rdbuf())) {}

    ~LoggerRedirect() { m_orig.rdbuf(m_old_buffer); }

    void flush(spdlog::level::level_enum level) {
        std::istringstream dumbtmp(ss.str());
        for (std::string line; std::getline(dumbtmp, line);)
            spdlog::log(level, line);
        ss.str("");
        ss.clear();
    }

  private:
    LoggerRedirect(const LoggerRedirect &);
    LoggerRedirect &operator=(const LoggerRedirect &);

    std::stringstream ss; ///< temporary object to redirect to for lifetime of this object
    std::ostream &m_orig; ///< Actual original ostream object (usually cout)
    std::streambuf *m_old_buffer; ///< pointer to original streambuf pointer (usually cout's)
};

}; // namespace utils

#endif
