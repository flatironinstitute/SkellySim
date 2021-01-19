#include <body.hpp>
#include <cnpy.hpp>
#include <kernels.hpp>
#include <parse_util.hpp>

void Body::build_preconditioner(double eta) {
    update_singularity_subtraction_vecs(eta);

    A_.resize(3 * num_nodes_ + 6, 3 * num_nodes_ + 6);
    A_.setZero();

    // M matrix
    A_.block(0, 0, 3 * num_nodes_, 3 * num_nodes_) =
        kernels::stresslet_times_normal(node_positions_, node_normals_, eta);

    for (int i = 0; i < num_nodes_; ++i) {
        A_.block(i * 3, 3 * i + 0, 3, 1) -= ex_.col(i) / node_weights_[i];
        A_.block(i * 3, 3 * i + 1, 3, 1) -= ey_.col(i) / node_weights_[i];
        A_.block(i * 3, 3 * i + 2, 3, 1) -= ez_.col(i) / node_weights_[i];
    }

    // K matrix
    for (int i = 0; i < num_nodes_; ++i) {
        // J matrix
        A_.block(i * 3, 3 * num_nodes_ + 0, 3, 3).diagonal().array() = -1.0;
        // rot matrix
        Eigen::Vector3d vec = node_positions_.col(i);
        A_.block(i * 3 + 0, 3 * num_nodes_, 1, 3) = -Eigen::Vector3d({0.0, vec[2], -vec[1]});
        A_.block(i * 3 + 1, 3 * num_nodes_, 1, 3) = -Eigen::Vector3d({-vec[2], 0.0, vec[0]});
        A_.block(i * 3 + 2, 3 * num_nodes_, 1, 3) = -Eigen::Vector3d({vec[1], -vec[0], 0.0});
    }

    // K^T matrix
    A_.block(3 * num_nodes_, 0, 6, 3 * num_nodes_) = A_.block(0, 3 * num_nodes_, 3 * num_nodes_, 6).transpose();

    // Last block is apparently diagonal.
    A_.block(3 * num_nodes_, 3 * num_nodes_, 6, 6).diagonal().array() = 1.0;

    A_LU_.compute(A_);
}

void Body::move(const Eigen::Vector3d &new_pos, const Eigen::Quaterniond &new_orientation) {
    position_ = new_pos;
    orientation_ = new_orientation;

    Eigen::Matrix3d rot = orientation_.toRotationMatrix();
    for (int i = 0; i < num_nodes_; ++i)
        node_positions_.col(i) = position_ + rot * node_positions_ref_.col(i);

    for (int i = 0; i < num_nodes_; ++i)
        node_normals_.col(i) = rot * node_normals_ref_.col(i);

}

void Body::update_singularity_subtraction_vecs(double eta) {
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(3, num_nodes_);

    e.row(0) = node_weights_.transpose();
    ex_ = kernels::stresslet_times_normal_times_density(node_positions_, node_normals_, e, eta);

    e.row(0).array() = 0.0;
    e.row(1) = node_weights_.transpose();
    ey_ = kernels::stresslet_times_normal_times_density(node_positions_, node_normals_, e, eta);

    e.row(1).array() = 0.0;
    e.row(2) = node_weights_.transpose();
    ez_ = kernels::stresslet_times_normal_times_density(node_positions_, node_normals_, e, eta);
}

void Body::load_precompute_data(const std::string &precompute_file) {
    cnpy::npz_t precomp = cnpy::npz_load(precompute_file);
    auto load_mat = [](cnpy::npz_t &npz, const char *var) {
        return Eigen::Map<Eigen::ArrayXXd>(npz[var].data<double>(), npz[var].shape[1], npz[var].shape[0]).matrix();
    };

    auto load_vec = [](cnpy::npz_t &npz, const char *var) {
        return Eigen::Map<Eigen::VectorXd>(npz[var].data<double>(), npz[var].shape[0]);
    };

    node_positions_ref_ = load_mat(precomp, "node_positions_ref");
    node_normals_ref_ = load_mat(precomp, "node_normals_ref");
    node_weights_ = load_vec(precomp, "node_weights");
}

Body::Body(const toml::table *body_table, const Params &params) {
    using namespace parse_util;
    using std::string;
    string precompute_file = parse_val_key<string>(body_table, "precompute_file");
    load_precompute_data(precompute_file);
    
    // TODO: add body assertions so that input file and precompute data necessarily agree
    num_nodes_ = node_positions_.cols();

    if (!!body_table->get("position"))
        position_ = parse_array_key<>(body_table, "position");

    if (!!body_table->get("orientation"))
        orientation_ = parse_array_key<Eigen::Quaterniond>(body_table, "orientation");

    move(position_, orientation_);
}
