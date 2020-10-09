#ifndef FIBER_HPP
#define FIBER_HPP
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

class Fiber {
  public:
    int num_points;
    double length;
    double bending_rigidity;
    typedef Eigen::MatrixXd matrix_t;
    typedef Eigen::ArrayXd array_t;
    matrix_t x;
    matrix_t xs;
    matrix_t xss;
    matrix_t xsss;
    matrix_t xssss;
    matrix_t stokeslet;

    typedef struct {
        array_t alpha;
        array_t alpha_roots;
        array_t alpha_tension;
        matrix_t D_1_0;
        matrix_t D_2_0;
        matrix_t D_3_0;
        matrix_t D_4_0;
        matrix_t P_X;
        matrix_t P_T;
        matrix_t P_cheb_representations_all_dof;
    } fib_mat_t;
    const static std::unordered_map<int, fib_mat_t> matrices;

    Fiber(int num_points, double bending_rigidity) : num_points(num_points), bending_rigidity(bending_rigidity) {
        this->x.resize(num_points, 3);
        this->xs.resize(num_points, 3);
        this->xss.resize(num_points, 3);
        this->xsss.resize(num_points, 3);
        this->xssss.resize(num_points, 3);
    }

    void update_derivatives();
    void update_stokeslet();
};

class FiberContainer {
  public:
    std::vector<Fiber> fibers;
    double slenderness_ratio;

    FiberContainer(int num_fibers, int num_points, double bending_rigidity) {
        fibers.reserve(num_fibers);
        for (int i = 0; i < num_fibers; ++i) {
            fibers.push_back(Fiber(num_points, bending_rigidity));
        }
    }

    void update_derivatives();
    void update_stokeslets();
};

#endif
