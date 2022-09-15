#ifndef STREAMLINE_HPP
#define STREAMLINE_HPP

#include <skelly_sim.hpp>

#include <fstream>

/// @brief Class representing a streamline
class StreamLine {
  public:
    Eigen::MatrixXd x; ///< Coordinates of streamline
    void compute();    ///< Compute streamline

    StreamLine(VectorRef &x0_, double tmax_, double precision_) : x(x0_), tmax(tmax_), precision(precision_){};

    void write(std::ofstream &ofs) {
        msgpack::pack(ofs, *this);
        ofs.flush();
    };
    MSGPACK_DEFINE_MAP(x);

  private:
    double tmax;
    double precision;
};

#endif
