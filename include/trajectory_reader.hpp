#ifndef TRAJECTORY_READER_HPP
#define TRAJECTORY_READER_HPP

#include <skelly_sim.hpp>

#include <string>

class TrajectoryReader {
  public:
    TrajectoryReader(const std::string &input_file, bool resume_flag);
    std::size_t read_next_frame();
    void unpack_current_frame(bool silence_output = false);

  private:
    int fd_;                    ///< File descriptor
    std::size_t buflen_;        ///< size of our file
    char *addr_;                ///< mmap address
    std::size_t offset_;        ///< current byte location in trajectory
    msgpack::object_handle oh_; ///< handle to last read frame
    bool resume_flag_;          ///< Reader being used to load resume data
};


#endif
