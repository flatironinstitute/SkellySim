#ifndef EIGEN_MATRIX_PLUGIN_H
#define EIGEN_MATRIX_PLUGIN_H

inline void msgpack_unpack(msgpack::object o) {
    if(o.type != msgpack::type::ARRAY) { throw msgpack::type_error(); }

    msgpack::object * p = o.via.array.ptr;

    std::string type;
    *p >> type;
    if (type != "__eigen__") { throw msgpack::type_error(); }

    size_t rows;
    size_t cols;

    ++p;
    *p >> rows;
    ++p;
    *p >> cols;
    this->resize(rows, cols);

    for (int i = 0; i < this->cols(); ++i) {
        for (int j = 0; j < this->rows(); ++j) {
            ++p;
            *p >> this->operator()(j, i);
        }
    }
}

template <typename Packer>
inline void msgpack_pack(Packer& pk) const {
    pk.pack_array(3 + this->rows()*this->cols());
    pk.pack(std::string("__eigen__"));
    pk.pack(this->rows());
    pk.pack(this->cols());

    for (int i = 0; i < this->cols(); ++i) {
        for (int j = 0; j < this->rows(); ++j) {
            pk.pack(this->operator()(j, i));
        }
    }
}

template <typename MSGPACK_OBJECT>
inline void msgpack_object(MSGPACK_OBJECT* o, msgpack::zone* z) const { }

#endif
