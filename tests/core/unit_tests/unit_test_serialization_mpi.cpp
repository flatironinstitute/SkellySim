
/// \file unit_test_serialization_mpi.cpp
/// \brief Unit tests for msgpack with serialization and MPI

// C++ includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// skelly includes
#include <msgpack.hpp>

// test files
#include "./mpi_environment.hpp"

// Test classes for what we are doing?
class Base {
  public:
    enum class MTYPE { DerivedA, DerivedB };

    MTYPE type_;

    virtual ~Base() = default;
};

MSGPACK_ADD_ENUM(Base::MTYPE);

class DerivedA : public Base {
  public:
    std::vector<double> x_;

    DerivedA() { type_ = Base::MTYPE::DerivedA; }

    MSGPACK_DEFINE(type_, x_);
};

class DerivedB : public Base {
  public:
    std::vector<int> x_;

    DerivedB() { type_ = Base::MTYPE::DerivedB; }

    MSGPACK_DEFINE(type_, x_);
};

struct DataContainer {
  public:
    std::unique_ptr<Base> base_ptr_;
    std::vector<double> test_double_;

    MSGPACK_DEFINE_MAP(base_ptr_, test_double_);
};

// Get the adaptors for the unique_pointers class?
namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
    namespace adaptor {
    // Class template specialization
    template <>
    struct convert<std::unique_ptr<Base>> {
        msgpack::object const &operator()(msgpack::object const &o, std::unique_ptr<Base> &v) const {

            // Sanity check
            if (o.type != msgpack::type::ARRAY) {
                std::cout << "ERROR: msgpack found " << (int)o.type << " rather than the ARRAY "
                          << (int)msgpack::type::ARRAY << std::endl;
                throw msgpack::type_error();
            }
            Base::MTYPE mtype;
            o.via.array.ptr[0].convert(mtype);

            switch (mtype) {
            case Base::MTYPE::DerivedA: {
                DerivedA derived_a;
                o.convert(derived_a);
                v = std::make_unique<DerivedA>(std::move(derived_a));
                break;
            }
            case Base::MTYPE::DerivedB: {
                DerivedB derived_b;
                o.convert(derived_b);
                v = std::make_unique<DerivedB>(std::move(derived_b));
                break;
            }
            default:
                throw msgpack::type_error();
            }
            return o;
        }
    };

    template <>
    struct pack<std::unique_ptr<Base>> {
        template <typename Stream>
        msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o, std::unique_ptr<Base> const &v) const {
            if (v->type_ == Base::MTYPE::DerivedA) {
                o.pack(*static_cast<DerivedA *>(v.get()));
            } else if (v->type_ == Base::MTYPE::DerivedB) {
                o.pack(*static_cast<DerivedB *>(v.get()));
            }

            return o;
        }
    };
    } // namespace adaptor
}
} // namespace msgpack

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

    return RUN_ALL_TESTS();
}

TEST(SerializationTest, MPIA) {

    // Get the rank and size
    int mrank = 0;
    int msize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
    MPI_Comm_size(MPI_COMM_WORLD, &msize);

    // Load the test properly
    DataContainer data;
    data.base_ptr_ = std::make_unique<DerivedA>();

    data.test_double_ = {3.14, 2.71, 1.62};

    // Allocate different numbers depending on the rank
    static_cast<DerivedA *>(data.base_ptr_.get())->x_ = {1.1 * mrank, 2.1 * mrank, 3.1 * mrank};

    // Make this look like the current implementation of the write command
    std::stringstream mergebuf;
    msgpack::pack(mergebuf, data);

    std::string msg_local = mergebuf.str();
    int msgsize_local = msg_local.size();
    std::vector<int> msgsize(msize);
    MPI_Gather(&msgsize_local, 1, MPI_INT, msgsize.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int msgsize_global = 0;
    std::vector<int> displs(msize + 1);
    for (int i = 0; i < msize; ++i) {
        msgsize_global += msgsize[i];
        displs[i + 1] = displs[i] + msgsize[i];
    }

    std::vector<uint8_t> msg = (mrank == 0) ? std::vector<uint8_t>(msgsize_global) : std::vector<uint8_t>();
    MPI_Gatherv(msg_local.data(), msgsize_local, MPI_CHAR, msg.data(), msgsize.data(), displs.data(), MPI_CHAR, 0,
                MPI_COMM_WORLD);

    if (mrank == 0) {
        msgpack::object_handle oh;
        std::size_t offset = 0;

        DataContainer to_write;
        for (int i = 0; i < msize; ++i) {
            msgpack::unpack(oh, (char *)msg.data(), msg.size(), offset);
            msgpack::object obj = oh.get();
            DataContainer const &min_state = obj.as<DataContainer>();

            const DerivedA *derived_a = static_cast<const DerivedA *>(min_state.base_ptr_.get());
            std::cout << "DerivedA values:";
            for (double val : derived_a->x_) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            if (i == 0) {
                EXPECT_EQ(0.0, derived_a->x_[0]);
                EXPECT_EQ(0.0, derived_a->x_[1]);
                EXPECT_EQ(0.0, derived_a->x_[2]);
            } else if (i == 2) {
                EXPECT_EQ(1.1, derived_a->x_[0]);
                EXPECT_EQ(2.1, derived_a->x_[1]);
                EXPECT_EQ(3.1, derived_a->x_[2]);
            }
        }
    }
}

