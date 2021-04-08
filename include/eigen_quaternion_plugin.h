#ifndef EIGEN_QUATERNION_PLUGIN_H
#define EIGEN_QUATERNION_PLUGIN_H

inline void msgpack_unpack(msgpack::object o) {
    if(o.type != msgpack::type::ARRAY) { throw msgpack::type_error(); }

    msgpack::object * p = o.via.array.ptr;

    std::string type;
    *p >> type;
    if (type != "__eigen__") { throw msgpack::type_error(); }

    ++p;
    *p >> this->w();
    ++p;
    *p >> this->x();
    ++p;
    *p >> this->y();
    ++p;
    *p >> this->z();
}

template <typename Packer>
inline void msgpack_pack(Packer& pk) const {
    pk.pack_array(5);
    pk.pack(std::string("__eigen__"));

    pk.pack(this->w());
    pk.pack(this->x());
    pk.pack(this->y());
    pk.pack(this->z());
}

template <typename MSGPACK_OBJECT>
inline void msgpack_object(MSGPACK_OBJECT* o, msgpack::zone* z) const { }

#endif
