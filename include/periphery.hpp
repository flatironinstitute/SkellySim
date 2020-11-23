#ifndef PERIPHERY_HPP
#define PERIPHERY_HPP

#include <iostream>
#include <Eigen/Core>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>

class Periphery {
    typedef Tpetra::MultiVector<> distributed_matrix_t;

  public:
    Periphery(const std::string &precompute_file);
    Teuchos::RCP<distributed_matrix_t> M_inv_;
    Teuchos::RCP<distributed_matrix_t> stresslet_;
};

#endif
