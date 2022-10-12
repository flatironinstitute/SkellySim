#ifndef STREAMLINE_HPP
#define STREAMLINE_HPP

#include <skelly_sim.hpp>

#include <fstream>

/// @brief Class representing a streamline
class StreamLine {
  public:
    Eigen::MatrixXd x;         ///< Coordinates of streamline
    Eigen::MatrixXd val;       ///< Velocities of streamline
    Eigen::VectorXd time;      ///< Evaluation times
    virtual void compute();    ///< Compute streamline

    StreamLine() = default;

    StreamLine(VectorRef &x0_, double dt_init_, double t_final_, double abs_err_, double rel_err_, bool back_integrate_)
        : x(x0_), dt_init(dt_init_), t_final(t_final_), abs_err(abs_err_), rel_err(rel_err_),
          back_integrate(back_integrate_) {
        compute();
    };

    void write(std::ofstream &ofs) {
        msgpack::pack(ofs, *this);
        ofs.flush();
    };
    MSGPACK_DEFINE_MAP(x, val, time);

  protected:
    double dt_init;
    double t_final;
    double abs_err;
    double rel_err;
    bool back_integrate;
};

class VortexLine : public StreamLine {
public:
    VortexLine() = default;

    VortexLine(VectorRef &x0_, double dt_init_, double t_final_, double abs_err_, double rel_err_,
               bool back_integrate_) {
        x = x0_;
        dt_init = dt_init_;
        t_final = t_final_;
        abs_err = abs_err_;
        rel_err = rel_err_;
        back_integrate = back_integrate_;
        compute();
    };

    virtual void compute() override; ///< Compute VortexLine
    MSGPACK_DEFINE_MAP(x, val, time);
};

#endif
