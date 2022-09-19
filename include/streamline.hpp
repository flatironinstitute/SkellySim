#ifndef STREAMLINE_HPP
#define STREAMLINE_HPP

#include <skelly_sim.hpp>

#include <fstream>

/// @brief Class representing a streamline
class StreamLine {
  public:
    Eigen::MatrixXd x; ///< Coordinates of streamline
    Eigen::MatrixXd v; ///< Velocities of streamline
    void compute();    ///< Compute streamline

    StreamLine() = default;

    StreamLine(VectorRef &x0_, double dt_init_, double t_final_, double abs_err_, double rel_err_)
        : x(x0_), dt_init(dt_init_), t_final(t_final_), abs_err(abs_err_), rel_err(rel_err_){
        compute();
    };

    void write(std::ofstream &ofs) {
        msgpack::pack(ofs, *this);
        ofs.flush();
    };
    MSGPACK_DEFINE_MAP(x, v);

  private:
    double dt_init;
    double t_final;
    double abs_err;
    double rel_err;
};

#endif
