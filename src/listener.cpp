#include <skelly_sim.hpp>

#include <listener.hpp>
#include <system.hpp>
#include <streamline.hpp>

#include <spdlog/spdlog.h>

#include <iostream>

typedef struct listener_command_t {
    std::size_t frame_no = 0;
    struct {
        double dt_init = 0.1;
        double t_final = 1.0;
        double abs_err = 1E-10;
        double rel_err = 1E-6;
        std::vector<double> x0;
        MSGPACK_DEFINE_MAP(dt_init, t_final, abs_err, rel_err, x0);
    } streamlines;

    MSGPACK_DEFINE_MAP(frame_no, streamlines);
} listener_command_t;

typedef struct listener_response_t {
    std::vector<StreamLine> streamlines;
    MSGPACK_DEFINE_MAP(streamlines);
} listener_response_t;


namespace listener {
void run() {
    spdlog::info("Entering listener mode...");
    uint64_t msgsize = 0;
    while (read(STDIN_FILENO, &msgsize, sizeof(msgsize)) > 0) {
        if (msgsize == 0)
            return;
        std::vector<char> data(msgsize);
        read(STDIN_FILENO, data.data(), msgsize);

        auto obj = msgpack::unpack(data.data(), msgsize).get().as<listener_command_t>();

        msgpack::sbuffer sbuf;
        msgpack::pack(sbuf, listener_response_t{.streamlines = {StreamLine(), StreamLine()}});

        uint64_t ressize = sbuf.size();
        spdlog::info("Returning buffer of size: {}", ressize);
        fwrite(&ressize, sizeof(uint64_t), 1, stdout);
        fwrite(sbuf.data(), 1, sbuf.size(), stdout);
        fflush(stdout);
    }
}
} // namespace listener
