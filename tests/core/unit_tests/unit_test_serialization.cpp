
/// \file unit_test_serialization.cpp
/// \brief Unit tests for serialization via msgpack (single MPI rank)

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

// This is a simple test to see if we can serialize a 'abstract base class' that isn't actually an ABC and
// serialize/deserilize it.
TEST(SerializationTest, SimpleA) {

    DataContainer data;
    data.base_ptr_ = std::make_unique<DerivedA>();

    data.test_double_ = {3.14, 2.71, 1.62};
    static_cast<DerivedA *>(data.base_ptr_.get())->x_ = {1.1, 2.2, 3.3};

    // Make this look like the current implementation of the write command
    std::stringstream sbuf;
    msgpack::pack(sbuf, data);
    auto const &mstr = sbuf.str();

    std::cout << "Successfully packed!\n";

    // Try to deserialize the data and deduce the type
    std::size_t offset = 0;
    msgpack::object_handle oh = msgpack::unpack(mstr.data(), mstr.size(), offset);
    msgpack::object obj = oh.get();

    // Check the object?
    std::cout << "Object: " << obj << std::endl;

    DataContainer data_deserialized;
    obj.convert(data_deserialized);

    // Can we deduce the type correctly?
    std::cout << "Deduced type: " << (int)data_deserialized.base_ptr_->type_ << std::endl;

    // Can we deserialize the information correctly?
    if (data_deserialized.base_ptr_->type_ == Base::MTYPE::DerivedA) {
        DerivedA *derived_a = static_cast<DerivedA *>(data_deserialized.base_ptr_.get());

        std::cout << "DerivedA values:\n";
        for (double val : derived_a->x_) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        EXPECT_EQ(derived_a->x_[0], 1.1);
        EXPECT_EQ(derived_a->x_[1], 2.2);
        EXPECT_EQ(derived_a->x_[2], 3.3);
    }
}

