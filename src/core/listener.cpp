#include <skelly_sim.hpp>

#include <listener.hpp>
#include <streamline.hpp>
#include <system.hpp>
#include <trajectory_reader.hpp>

namespace listener {
typedef struct listener_command_t {
    std::size_t frame_no = 0;
    std::string evaluator = "CPU";

    struct streamlines {
        double dt_init = 0.1;
        double t_final = 1.0;
        double abs_err = 1E-10;
        double rel_err = 1E-6;
        bool back_integrate = true;
        Eigen::MatrixXd x0;
        MSGPACK_DEFINE_MAP(dt_init, t_final, abs_err, rel_err, back_integrate, x0);
    } streamlines;

    struct vortexlines {
        double dt_init = 0.1;
        double t_final = 1.0;
        double abs_err = 1E-10;
        double rel_err = 1E-6;
        bool back_integrate = true;
        Eigen::MatrixXd x0;
        MSGPACK_DEFINE_MAP(dt_init, t_final, abs_err, rel_err, back_integrate, x0);
    } vortexlines;

    struct velocity_field {
        Eigen::MatrixXd x;
        MSGPACK_DEFINE_MAP(x);
    } velocity_field;

    MSGPACK_DEFINE_MAP(frame_no, evaluator, streamlines, vortexlines, velocity_field);
} listener_command_t;

typedef struct listener_response_t {
    double time;
    std::size_t i_frame;
    std::size_t n_frames;
    std::vector<StreamLine> streamlines;
    std::vector<VortexLine> vortexlines;
    Eigen::MatrixXd velocity_field;
    MSGPACK_DEFINE_MAP(time, i_frame, n_frames, streamlines, vortexlines, velocity_field);
} listener_response_t;

std::vector<StreamLine> process_streamlines(struct listener_command_t::streamlines &request) {
    if (request.x0.cols())
        spdlog::info("Processing {} streamlines", request.x0.cols());

    std::vector<StreamLine> streamlines;

    for (int i = 0; i < request.x0.cols(); i++)
        streamlines.push_back(StreamLine(request.x0.col(i), request.dt_init, request.t_final, request.abs_err,
                                         request.rel_err, request.back_integrate));

    return streamlines;
}

std::vector<VortexLine> process_vortexlines(struct listener_command_t::vortexlines &request) {
    if (request.x0.cols())
        spdlog::info("Processing {} vortex lines", request.x0.cols());

    std::vector<VortexLine> vortexlines;

    for (int i = 0; i < request.x0.cols(); i++)
        vortexlines.push_back(VortexLine(request.x0.col(i), request.dt_init, request.t_final, request.abs_err,
                                         request.rel_err, request.back_integrate));

    return vortexlines;
}

Eigen::MatrixXd process_velocity_field(struct listener_command_t::velocity_field &request) {
    if (request.x.cols())
        spdlog::info("Processing {} velocity field points", request.x.cols());

    if (!request.x.size())
        return Eigen::MatrixXd();
    return System::velocity_at_targets(request.x);
}

void run() {
    spdlog::info("Entering listener mode...");
    TrajectoryReader traj("skelly_sim.out", false);

    uint64_t msgsize = 0;
    while (read(STDIN_FILENO, &msgsize, sizeof(msgsize))) {
        if (msgsize == 0) {
            spdlog::info("Terminate message received. Exiting listener mode");
            return;
        }

        std::vector<char> cmd_payload(msgsize);
        ssize_t bytes_read = 0;
        while (bytes_read < msgsize) {
            ssize_t readsize = read(STDIN_FILENO, cmd_payload.data() + bytes_read, msgsize - bytes_read);
            if (readsize < 0) {
                spdlog::error("Error reading payload");
                return;
            }
            bytes_read += readsize;
        }

        auto cmd = msgpack::unpack(cmd_payload.data(), msgsize).get().as<listener_command_t>();

        // Load frame and ignore command if frame invalid
        if (traj.load_frame(cmd.frame_no)) {
            uint64_t msgsize = 0;
            fwrite(&msgsize, sizeof(msgsize), 1, stdout);
            fflush(stdout);
            continue;
        }
        System::set_evaluator(cmd.evaluator);

        listener_response_t response{
            .time = System::get_properties().time,
            .i_frame = cmd.frame_no,
            .n_frames = traj.get_n_frames(),
            .streamlines = process_streamlines(cmd.streamlines),
            .vortexlines = process_vortexlines(cmd.vortexlines),
            .velocity_field = process_velocity_field(cmd.velocity_field),
        };
        msgpack::sbuffer sbuf;
        msgpack::pack(sbuf, response);

        uint64_t ressize = sbuf.size();
        spdlog::info("Returning buffer of size: {}", ressize);
        fwrite(&ressize, sizeof(ressize), 1, stdout);
        fwrite(sbuf.data(), 1, sbuf.size(), stdout);
        fflush(stdout);
    }
}
} // namespace listener
