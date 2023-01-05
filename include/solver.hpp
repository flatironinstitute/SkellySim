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
    bool solve();
    void apply_preconditioner();
    CVectorMap get_solution() { return CVectorMap(X_->getData(0).getRawPtr(), X_->getLocalLength()); };
    double get_residual() {
        Teuchos::RCP<SV> Y(new SV(map_));
        matvec_->apply(*X_, *Y);
        CVectorMap RHS_map(RHS_->getData(0).getRawPtr(), RHS_->getLocalLength());
        CVectorMap Y_map(Y->getData(0).getRawPtr(), Y->getLocalLength());
        double residual = (RHS_map - Y_map).squaredNorm();
        MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return sqrt(residual);
    }

  private:
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    Teuchos::RCP<precond_T> preconditioner_;
    Teuchos::RCP<matvec_T> matvec_;
    Teuchos::RCP<SV> X_;
    Teuchos::RCP<SV> RHS_;
    Teuchos::RCP<const Tpetra::Map<>> map_;
};

#endif
