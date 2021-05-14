#ifndef BODY_HPP
#define BODY_HPP

#include <skelly_sim.hpp>

#include <Eigen/LU>
#include <kernels.hpp>
#include <params.hpp>

class Periphery;
class SphericalBody;
class DeformableBody;

/// Class for "small" bodies such as MTOCs
class Body {
  public:
    int n_nodes_; ///< Number of nodes representing the body surface

    Eigen::VectorXd RHS_;                ///< Current 'right-hand-side' for matrix formulation of solver
    Eigen::MatrixXd node_positions_;     ///< [ 3 x n_nodes ] node positions in lab frame
    Eigen::MatrixXd node_positions_ref_; ///< [ 3 x n_nodes ] node positions in reference 'body' frame
    Eigen::MatrixXd node_normals_;       ///< [ 3 x n_nodes ] node normals in lab frame
    Eigen::MatrixXd node_normals_ref_;   ///< [ 3 x n_nodes ] node normals in reference 'body' frame
    Eigen::VectorXd node_weights_;       ///< [ n_nodes ] far field quadrature weights for nodes

    /// [ 3 x n_nucleation_sites ] nucleation site positions in reference 'body' frame
    Eigen::MatrixXd nucleation_sites_ref_;
    /// [ 3 x n_nucleation_sites ] nucleation site positions in lab frame
    Eigen::MatrixXd nucleation_sites_;

    Body(const toml::value &body_table, const Params &params);
    Body() = default; ///< default constructor...

    virtual void update_RHS(MatrixRef &v_on_body);
    virtual void update_cache_variables(double eta);
    virtual void update_preconditioner(double eta);
    virtual void load_precompute_data(const std::string &input_file);
    virtual void step(double dt, VectorRef &body_solution);

    virtual int get_solution_size() const;
    virtual Eigen::VectorXd matvec(MatrixRef &v_bodies, VectorRef &body_solution) const;
    virtual Eigen::VectorXd apply_preconditioner(VectorRef &x) const;

    /// @brief Make a copy of this instance
    virtual std::unique_ptr<Body> clone() const { return std::make_unique<Body>(*this); }

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const Periphery &periphery, double threshold) const {
        // FIXME: there is probably a way to make our objects abstract base classes, but it makes the containers weep if
        // you make this a pure virtual function, so instead we just throw an error.
        throw std::runtime_error("Collision undefined on base Body class\n");
    };

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const Body &body, double threshold) const {
        throw std::runtime_error("Collision undefined on base Body class\n");
    };

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const SphericalBody &body, double threshold) const {
        throw std::runtime_error("Collision undefined on base Body class\n");
    };

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const DeformableBody &body, double threshold) const {
        throw std::runtime_error("Collision undefined on base Body class\n");
    };

    /// For structures with fixed size Eigen::Vector types, this ensures alignment if the
    /// structure is allocated via `new`
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Container for multiple generic Body objects
class BodyContainer {
  private:
    int world_rank_ = 0; ///< MPI world rank
    int world_size_;     ///< MPI world size

  public:
    /// Vector of body pointers
    std::vector<std::unique_ptr<Body>> bodies;

    /// Pointer to FMM stresslet kernel (Stokes)
    std::shared_ptr<kernels::FMM<stkfmm::Stk3DFMM>> stresslet_kernel_;
    /// Pointer to FMM oseen kernel (PVel)
    std::shared_ptr<kernels::FMM<stkfmm::Stk3DFMM>> oseen_kernel_;

    /// @brief Empty container constructor to avoid initialization list complications.
    BodyContainer() = default;
    BodyContainer(toml::array &body_tables, Params &params);

    // FIXME: remove redundant code in =/copy
    /// @brief Copy constructor...
    BodyContainer(const BodyContainer &orig) {
        for (auto &body : orig.bodies) {
            bodies.push_back(body->clone());
        }
        world_rank_ = orig.world_rank_;
        world_size_ = orig.world_size_;
        stresslet_kernel_ = orig.stresslet_kernel_;
        oseen_kernel_ = orig.oseen_kernel_;
    };

    /// @brief Assignment operator...
    BodyContainer &operator=(const BodyContainer orig) {
        bodies.clear();
        for (auto &body : orig.bodies) {
            bodies.push_back(body->clone());
        }
        world_rank_ = orig.world_rank_;
        world_size_ = orig.world_size_;
        stresslet_kernel_ = orig.stresslet_kernel_;
        oseen_kernel_ = orig.oseen_kernel_;
        return *this;
    }

    /// @brief Get total number of nodes associated with body
    ///
    /// See BodyContainer::get_local_solution_size for notes about how this is distributed
    int get_local_node_count() const {
        if (world_rank_ != 0)
            return 0;

        int tot = 0;
        for (const auto &body : bodies)
            tot += body->n_nodes_;
        return tot;
    }

    /// @brief Get the size of all bodies contribution to the matrix problem solution, regardless of rank
    int get_global_solution_size() const {
        int sol_size = 0;
        for (const auto &body : bodies)
            sol_size += body->get_solution_size();
        return sol_size;
    }

    /// @brief Get the size of all local bodies contribution to the matrix problem solution
    ///
    /// Since there aren't many bodies, and there is no easy way to synchronize them across processes, the rank 0
    /// process handles all components of the solution.
    int get_local_solution_size() const { return world_rank_ == 0 ? get_global_solution_size() : 0; }

    void update_RHS(MatrixRef &v_on_body);
    Eigen::VectorXd get_RHS() const;
    Eigen::MatrixXd get_global_node_positions() const;
    Eigen::MatrixXd get_local_node_positions() const;
    Eigen::MatrixXd get_local_node_normals() const;

    Eigen::VectorXd matvec(MatrixRef &v_bodies, VectorRef &x_bodies) const;
    Eigen::VectorXd apply_preconditioner(VectorRef &X) const;
    Eigen::MatrixXd flow(MatrixRef &r_trg, VectorRef &body_solutions, double eta) const;
    Eigen::MatrixXd flow_body_spherical(const std::vector<const Body *> &bodies, MatrixRef &r_trg,
                                        VectorRef &body_solution, double eta) const;
    Eigen::MatrixXd flow_body_deformable(const std::vector<const Body *> &bodies, MatrixRef &r_trg,
                                         VectorRef &body_solution, double eta) const;
    void step(VectorRef &body_sol, double dt) const;

    /// @brief Update cache variables for each Body. @see Body::update_cache_variables
    void update_cache_variables(double eta) {
        for (auto &body : bodies)
            body->update_cache_variables(eta);
    }

    /// @brief Get copy of a given nucleation site
    ///
    /// @param[in] i_body index of relevant body
    /// @param[in] j_site index of nucleation site
    Eigen::Vector3d get_nucleation_site(int i_body, int j_site) const {
        return bodies[i_body]->nucleation_sites_.col(j_site);
    };

    /// @brief get reference to ith Body without inspecting internal BodyContainer::bodies
    ///
    /// @param[in] i index of body in container
    /// @return Reference to ith Body in bodies
    const Body &at(size_t i) const { return *bodies.at(i); };

    /// @brief return number of bodies relevant for local calculations
    size_t get_local_count() const { return (world_rank_ == 0) ? bodies.size() : 0; };
    /// @brief return number of bodies relevant for global calculations
    size_t get_global_count() const { return bodies.size(); };

    /// @brief Return total number of body nodes across all MPI ranks
    size_t get_global_node_count() const {
        size_t n_nodes = 0;
        for (const auto &body : bodies)
            n_nodes += body->n_nodes_;
        return n_nodes;
    }

    /// @brief Return total number of body nucleation sites across all MPI ranks
    size_t get_global_site_count() const {
        size_t n_sites = 0;
        for (const auto &body : bodies)
            n_sites += body->nucleation_sites_.cols();
        return n_sites;
    }

    /// @brief msgpack serialization routine
    MSGPACK_DEFINE(bodies);
};

/// @brief Spherical Body...
class SphericalBody : public Body {
  public:
    /// @brief Construct spherical body. @see Body
    /// @param[in] body_table Parsed TOML body table. Must have 'radius' key defined.
    /// @param[in] params Initialized Params object
    SphericalBody(const toml::value &body_table, const Params &params);

    /// Duplicate SphericalBody object
    std::unique_ptr<Body> clone() const override { return std::make_unique<SphericalBody>(*this); };

    // Parameters unique to spherical body
    double radius_;                  ///< Radius of body
    Eigen::Vector3d position_;       ///< Instantaneous lab frame position of body, usually the centroid
    Eigen::Quaterniond orientation_; ///< Instantaneous orientation of body
    Eigen::Quaterniond orientation_ref_ = {1.0, 0.0, 0.0, 0.0}; ///< Reference orientation of body
    Eigen::Vector3d velocity_;                                  ///<  Net instantaneous lab frame velocity of body
    Eigen::Vector3d angular_velocity_;         ///< Net instantaneous lab frame angular velocity of body
    Eigen::Matrix<double, 6, 1> force_torque_; ///< Net force+torque vector [fx,fy,fz,tx,ty,tz] about centroid

    Eigen::MatrixXd ex_; ///< [ 3 x num_nodes ] Singularity subtraction vector along x
    Eigen::MatrixXd ey_; ///< [ 3 x num_nodes ] Singularity subtraction vector along y
    Eigen::MatrixXd ez_; ///< [ 3 x num_nodes ] Singularity subtraction vector along z
    Eigen::MatrixXd K_;  ///< [ 3*num_nodes x 6 ] matrix that helps translate body info to nodes
    Eigen::Vector3d external_force_{0.0, 0.0, 0.0}; ///< [3] vector of constant external force on body in lab frame

    Eigen::MatrixXd A_;                         ///< Matrix representation of body for solver
    Eigen::PartialPivLU<Eigen::MatrixXd> A_LU_; ///< LU decomposition of A_ for preconditioner

    /// Return reference to body COM position
    void update_RHS(MatrixRef &v_on_body) override;
    void update_cache_variables(double eta) override;
    void update_preconditioner(double eta) override;
    void load_precompute_data(const std::string &input_file) override;
    void step(double dt, VectorRef &body_solution) override;

    int get_solution_size() const override { return n_nodes_ * 3 + 6; };
    Eigen::VectorXd matvec(MatrixRef &v_bodies, VectorRef &body_solution) const override;

    const Eigen::Vector3d &get_position() const { return position_; };
    void move(const Eigen::Vector3d &new_pos, const Eigen::Quaterniond &new_orientation);
    void update_K_matrix();
    void update_singularity_subtraction_vecs(double eta);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> unpack_solution_vector(VectorRef &x) const;

    bool check_collision(const Periphery &periphery, double threshold) const override;
    bool check_collision(const Body &body, double threshold) const override;
    bool check_collision(const SphericalBody &body, double threshold) const override;
    bool check_collision(const DeformableBody &body, double threshold) const override;

    /// @brief Serialize body automatically with msgpack macros
    MSGPACK_DEFINE_MAP(position_, orientation_);
};

/// @brief Spherical Body...
class DeformableBody : public Body {
  public:
    /// @brief Construct spherical body. @see Body
    /// @param[in] body_table Parsed TOML body table. Must have 'radius' key defined.
    /// @param[in] params Initialized Params object
    DeformableBody(const toml::value &body_table, const Params &params) : Body(body_table, params){};

    /// Duplicate SphericalBody object
    std::unique_ptr<Body> clone() const override { return std::make_unique<DeformableBody>(*this); };

    void update_RHS(MatrixRef &v_on_body) override;
    void update_cache_variables(double eta) override;
    void update_preconditioner(double eta) override;
    void load_precompute_data(const std::string &input_file) override;
    void step(double dt, VectorRef &body_solution) override;
    int get_solution_size() const override { return n_nodes_ * 4; };
    Eigen::VectorXd matvec(MatrixRef &v_bodies, VectorRef &body_solution) const override;

    bool check_collision(const Periphery &periphery, double threshold) const override;
    bool check_collision(const Body &body, double threshold) const override;
    bool check_collision(const SphericalBody &body, double threshold) const override;
    bool check_collision(const DeformableBody &body, double threshold) const override;
    MSGPACK_DEFINE_MAP(node_positions_, node_normals_);
};

#endif
