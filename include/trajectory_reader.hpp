#ifndef TRAJECTORY_READER_HPP
#define TRAJECTORY_READER_HPP

#include <skelly_sim.hpp>

#include <string>

class TrajectoryReader {
  public:
    TrajectoryReader(const std::string &input_file, bool resume_flag);
    std::size_t read_next_frame();
    bool load_frame(std::size_t frameno);
    void unpack_current_frame(bool silence_output = false);
    std::size_t get_n_frames() const { return index.offsets.size(); }

  private:
    int fd_;                           ///< File descriptor
    std::size_t buflen_;               ///< size of our file
    char *addr_;                       ///< mmap address
    std::size_t offset_;               ///< current byte location in trajectory
    msgpack::object_handle oh_;        ///< handle to last read frame
    bool resume_flag_;                 ///< Reader being used to load resume data
    long int mtime;

    struct {
        long int mtime;
        std::vector<std::size_t> offsets; ///< Vector of offsets for each frame
        std::vector<double> times;        ///< Vector of times for each frame
        MSGPACK_DEFINE_MAP(mtime, offsets, times);
    } index;

    void load_index(const std::string &traj_file);
    void build_index(const std::string &index_file);
};

#endif
