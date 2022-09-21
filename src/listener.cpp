#include <skelly_sim.hpp>

#include <listener.hpp>
#include <streamline.hpp>
#include <system.hpp>
#include <trajectory_reader.hpp>

#include <spdlog/spdlog.h>

#include <cstdint>
#include <iostream>

namespace listener {

typedef struct listener_command_t {
    std::size_t frame_no = 0;
    struct streamlines {
        double dt_init = 0.1;
        double t_final = 1.0;
        double abs_err = 1E-10;
        double rel_err = 1E-6;
        Eigen::MatrixXd x0;
        MSGPACK_DEFINE_MAP(dt_init, t_final, abs_err, rel_err, x0);
    } streamlines;

    struct velocity_field {
        Eigen::MatrixXd x;
        MSGPACK_DEFINE_MAP(x);
    } velocity_field;

    MSGPACK_DEFINE_MAP(frame_no, streamlines, velocity_field);
} listener_command_t;

typedef struct listener_response_t {
    double time;
    std::size_t i_frame;
    std::size_t n_frames;
    std::vector<StreamLine> streamlines;
    Eigen::MatrixXd velocity_field;
    MSGPACK_DEFINE_MAP(time, i_frame, n_frames, streamlines, velocity_field);
} listener_response_t;

std::vector<StreamLine> process_streamlines(struct listener_command_t::streamlines &request) {
    std::vector<StreamLine> streamlines;

    for (int i = 0; i < request.x0.cols(); i++)
        streamlines.push_back(StreamLine(request.x0.col(i), request.dt_init, request.t_final, request.abs_err, request.rel_err));

    return streamlines;
}

Eigen::MatrixXd process_velocity_field(struct listener_command_t::velocity_field &request) {
    if (!request.x.size())
        return Eigen::MatrixXd();
    return System::velocity_at_targets(request.x);
}

void run() {
    spdlog::info("Entering listener mode...");
    TrajectoryReader traj("skelly_sim.out", false);

    uint64_t msgsize = 0;
    while (read(STDIN_FILENO, &msgsize, sizeof(msgsize)) > 0) {
        if (msgsize == 0)
            return;
        std::vector<char> cmd_payload(msgsize);
        read(STDIN_FILENO, cmd_payload.data(), msgsize);

        auto cmd = msgpack::unpack(cmd_payload.data(), msgsize).get().as<listener_command_t>();

        // Load frame and ignore command if frame invalid
        if (traj.load_frame(cmd.frame_no)) {
            uint64_t msgsize = 0;
            fwrite(&msgsize, sizeof(msgsize), 1, stdout);
            fflush(stdout);
            continue;
        }

        listener_response_t response{
            .time = System::get_properties().time,
            .i_frame = cmd.frame_no,
            .n_frames = traj.get_n_frames(),
            .streamlines = process_streamlines(cmd.streamlines),
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
