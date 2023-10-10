#ifndef SERIALIZATION_HPP_
#define SERIALIZATION_HPP_

#include <fiber_container_base.hpp>
#include <fiber_container_finitedifference.hpp>
#include <fiber_finitedifference.hpp>

#include <msgpack.hpp>

/// Serialization and deserialization routines for msgpack
///
/// FiberContainerBase NOT implemented (screws up everything)
/// FiberContainerFinitedifference implemented

/// @brief Custom serializaiton and deserialization of the unique_ptr for FiberContainerBase
namespace msgpack {

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {

    namespace adaptor {

    template <>
    struct convert<std::unique_ptr<FiberContainerBase>> {
        msgpack::object const &operator()(msgpack::object const &o, std::unique_ptr<FiberContainerBase> &v) const {
            // Sanity check on the contents of the unique pointer
            if (o.type != msgpack::type::ARRAY) {
                throw msgpack::type_error();
            }

            FiberContainerBase::FIBERTYPE fiber_type;
            o.via.array.ptr[0].convert(fiber_type);

            switch (fiber_type) {
            case FiberContainerBase::FIBERTYPE::FiniteDifference: {
                FiberContainerFinitedifference fc_finitediff;
                o.convert(fc_finitediff);
                v = std::make_unique<FiberContainerFinitedifference>(std::move(fc_finitediff));
                break;
            }
            default:
                throw msgpack::type_error();
            } // switch
            return o;
        }
    };

    template <>
    struct pack<std::unique_ptr<FiberContainerBase>> {
        template <typename Stream>
        msgpack::packer<Stream> &operator()(msgpack::packer<Stream> &o,
                                            std::unique_ptr<FiberContainerBase> const &v) const {
            if (v->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
                o.pack(*static_cast<FiberContainerFinitedifference *>(v.get()));
            }

            return o;
        }
    };

    } // namespace adaptor
} // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)

} // namespace msgpack

#endif

