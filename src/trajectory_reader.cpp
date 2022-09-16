#include <trajectory_reader.hpp>

#include <body.hpp>
#include <fiber.hpp>
#include <io_maps.hpp>
#include <periphery.hpp>
#include <rng.hpp>
#include <system.hpp>

#include <fstream>

#include <sys/mman.h>
#include <spdlog/spdlog.h>
#include <mpi.h>

TrajectoryReader::TrajectoryReader(const std::string &input_file, bool resume_flag) : offset_(0), resume_flag_(resume_flag) {
    fd_ = open(input_file.c_str(), O_RDONLY);
    if (fd_ == -1)
        throw std::runtime_error("Unable to open trajectory file " + input_file + " for resume.");

    struct stat sb;
    if (fstat(fd_, &sb) == -1)
        throw std::runtime_error("Error statting " + input_file + " for resume.");

    buflen_ = sb.st_size;

    addr_ = static_cast<char *>(mmap(NULL, buflen_, PROT_READ, MAP_PRIVATE, fd_, 0u));
    if (addr_ == MAP_FAILED)
        throw std::runtime_error("Error mapping " + input_file + " for resume.");
}

std::size_t TrajectoryReader::TrajectoryReader::read_next_frame() {
    if (offset_ >= buflen_)
        return 0;

    std::size_t old_offset = offset_;
    msgpack::unpack(oh_, addr_, buflen_, offset_);
    return offset_ - old_offset;
}


void TrajectoryReader::unpack_current_frame(bool silence_output) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    msgpack::object obj = oh_.get();

    input_map_t const &min_state = obj.as<input_map_t>();

    auto &params = *System::get_params();
    auto &properties = System::get_properties();
    auto &fc = *System::get_fiber_container();
    auto &bc = *System::get_body_container();
    auto &shell = *System::get_shell();

    const int n_fib_tot = min_state.fibers.fibers.size();
    const int fib_count_big = n_fib_tot / size + 1;
    const int fib_count_small = n_fib_tot / size;
    const int n_fib_big = n_fib_tot % size;

    std::vector<int> counts(size);
    std::vector<int> displs(size + 1);
    for (int i = 0; i < size; ++i) {
        counts[i] = ((i < n_fib_big) ? fib_count_big : fib_count_small);
        displs[i + 1] = displs[i] + counts[i];
    }

    properties.time = min_state.time;
    properties.dt = min_state.dt;
    std::vector<bool> is_minus_clamped;
    // FIXME: Hack to work around not saving clamp state
    if (!params.dynamic_instability.n_nodes) {
        for (auto &fib : fc.fibers)
            is_minus_clamped.push_back(fib.is_minus_clamped());
    } else {
        throw std::runtime_error("Resume is broken in this version of SkellySim with dynamic instability. :(");
    }

    fc.fibers.clear();
    int i_fib = 0;
    for (const auto &min_fib : min_state.fibers.fibers) {
        if (i_fib >= displs[rank] && i_fib < displs[rank + 1]) {
            fc.fibers.emplace_back(Fiber(min_fib, params.eta));
            fc.fibers.back().minus_clamped_ = is_minus_clamped[i_fib - displs[rank]];
        }
        i_fib++;
    }

    // make sure sublist pointers are initialized, and then fill them in
    bc.populate_sublists();
    for (int i = 0; i < bc.spherical_bodies.size(); ++i)
        bc.spherical_bodies[i]->min_copy(min_state.bodies.spherical_bodies[i]);
    for (int i = 0; i < bc.deformable_bodies.size(); ++i)
        bc.deformable_bodies[i]->min_copy(min_state.bodies.deformable_bodies[i]);
    if (size > min_state.rng_state.size() && resume_flag_) {
        spdlog::error(
            "More MPI ranks provided than previous run for resume. This is currently unsupported for RNG reasons.");
        MPI_Finalize();
        exit(1);
    } else if (size < min_state.rng_state.size() && !silence_output && resume_flag_) {
        spdlog::warn("Fewer MPI ranks provided than previous run for resume. This will be non-deterministic if using "
                     "the RNG.");
    }
    if (size != min_state.rng_state.size() && resume_flag_) {
        spdlog::error("Different number MPI ranks provided than previous run for resume. This is currently broken.");
        MPI_Finalize();
        exit(1);
    }

    RNG::init(min_state.rng_state[rank]);

    if (shell.is_active())
        shell.solution_vec_ = min_state.shell.solution_vec_.segment(shell.node_displs_[rank], shell.node_counts_[rank]);

    auto [fiber_sol, shell_sol, body_sol] = System::get_solution_maps(System::get_curr_solution().data());
    shell_sol = shell.solution_vec_;

    if (rank == 0) {
        std::size_t body_offset = 0;
        for (auto &body : bc.bodies) {
            body_sol.segment(body_offset, body->solution_vec_.size()) = body->solution_vec_;
            body_offset += body->solution_vec_.size();
        }
    }

    std::size_t fiber_offset = 0;
    for (auto &fib : fc.fibers) {
        for (int i = 0; i < 3; ++i) {
            fiber_sol.segment(fiber_offset, fib.n_nodes_) = fib.x_.row(i);
            fiber_offset += fib.n_nodes_;
        }
        fiber_sol.segment(fiber_offset, fib.n_nodes_) = fib.tension_;
        fiber_offset += fib.n_nodes_;
    }

    fc.update_cache_variables(properties.dt, params.eta);
    bc.update_cache_variables(params.eta);
}
