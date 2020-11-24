#include <iostream>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <Tpetra_Core.hpp>
#include <Teuchos_Comm.hpp>
#include <periphery.hpp>

int main(int argc, char *argv[]) {
    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();

    try {
        Periphery sphere("test_periphery.npz");
    } catch (std::runtime_error e) {
        std::cout << e.what() << std::endl;
        std::cout << "Test failed" << std::endl;
        return 1;
    }
    if (rank == 0) {
        std::cout << "Test passed\n";
    }

    return 0;
}
