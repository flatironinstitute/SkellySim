#include <skelly_sim.hpp>

#include <solver.hpp>

#include <Tpetra_Import.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>

class P_inv_hydro : public Tpetra::Operator<> {
  public:
    // Tpetra::Operator subclasses should always define these four typedefs.
    typedef Tpetra::Operator<>::scalar_type scalar_type;
    typedef Tpetra::Operator<>::local_ordinal_type local_ordinal_type;
    typedef Tpetra::Operator<>::global_ordinal_type global_ordinal_type;
    typedef Tpetra::Operator<>::node_type node_type;
    // The type of the input and output arguments of apply().
    typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> MV;
    // The Map specialization used by this class.
    typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;

  private:
    // This is an implementation detail; users don't need to see it.
    typedef Tpetra::Import<local_ordinal_type, global_ordinal_type, node_type> import_type;

  public:
    // comm: The communicator over which to distribute those rows and columns.
    P_inv_hydro(const Teuchos::RCP<const Teuchos::Comm<int>> comm);
    virtual ~P_inv_hydro() {}

    Teuchos::RCP<const map_type> getDomainMap() const { return opMap_; };
    Teuchos::RCP<const map_type> getRangeMap() const { return opMap_; };

    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const;

  private:
    Teuchos::RCP<const map_type> opMap_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    const int rank_;
};

class A_fiber_hydro : public Tpetra::Operator<> {
  public:
    // Tpetra::Operator subclasses should always define these four typedefs.
    typedef Tpetra::Operator<>::scalar_type scalar_type;
    typedef Tpetra::Operator<>::local_ordinal_type local_ordinal_type;
    typedef Tpetra::Operator<>::global_ordinal_type global_ordinal_type;
    typedef Tpetra::Operator<>::node_type node_type;
    // The type of the input and output arguments of apply().
    typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> MV;
    // The Map specialization used by this class.
    typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;

  private:
    // This is an implementation detail; users don't need to see it.
    typedef Tpetra::Import<local_ordinal_type, global_ordinal_type, node_type> import_type;

  public:
    // comm: The communicator over which to distribute those rows and columns.
    A_fiber_hydro(const Teuchos::RCP<const Teuchos::Comm<int>> comm);
    virtual ~A_fiber_hydro() {}
    Teuchos::RCP<const map_type> getDomainMap() const { return opMap_; };
    Teuchos::RCP<const map_type> getRangeMap() const { return opMap_; };

    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const;

  private:
    Teuchos::RCP<const map_type> opMap_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    const int rank_;
};
