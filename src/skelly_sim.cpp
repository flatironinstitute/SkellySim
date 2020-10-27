#include <Teuchos_CommandLineProcessor.hpp>
#include <Tpetra_Core.hpp>

using namespace Teuchos;

int main(int argc, char *argv[]) {

    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    {
        RCP<const Comm<int>> comm = Tpetra::getDefaultComm();
        const int rank = comm->getRank();
        const int size = comm->getSize();

        CommandLineProcessor cmdp(false, true);
    }

    return 0;
}
