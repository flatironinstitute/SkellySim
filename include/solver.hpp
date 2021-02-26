#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <skelly_sim.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

template <typename precond_T, typename matvec_T>
class Solver {
  public:
    typedef typename matvec_T::scalar_type ST;
    typedef Tpetra::Vector<typename matvec_T::scalar_type, typename matvec_T::local_ordinal_type,
                           typename matvec_T::global_ordinal_type, typename matvec_T::node_type>
        SV;
    typedef typename matvec_T::MV MV;
    typedef Tpetra::Operator<ST> OP;

    Solver() {
        comm_ = Tpetra::getDefaultComm();
        matvec_ = rcp(new matvec_T(comm_));
        preconditioner_ = rcp(new precond_T(comm_));
        map_ = matvec_->getDomainMap();
        X_ = rcp(new SV(map_));
        RHS_ = rcp(new SV(map_));
    };
    void set_RHS();
    void solve();
    void apply_preconditioner();

  private:
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    Teuchos::RCP<precond_T> preconditioner_;
    Teuchos::RCP<matvec_T> matvec_;
    Teuchos::RCP<SV> X_;
    Teuchos::RCP<SV> RHS_;
    // Teuchos::RCP<vec_type> res_;
    Teuchos::RCP<const Tpetra::Map<>> map_;
};

#endif
