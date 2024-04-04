
/// \file unit_test_fiber_base.cpp
/// \brief Unit tests for FiberChebyshevPenalty class

// C++ includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// skelly includes
#include <fiber_chebyshev_penalty_autodiff.hpp>
#include <fiber_chebyshev_penalty_autodiff_external.hpp>
#include <msgpack.hpp>

// test files
#include "./julia_fiber_penalty_results.hpp"
#include "./mpi_environment.hpp"

// Make it a little easier to get the class calls
using namespace skelly_fiber;

// Test the penalty version constructor
TEST(FiberChebysehvPenaltyAutodiff, real_constructor) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);

    // Now create our fiber in XYT
    Eigen::VectorXd init_X = Eigen::VectorXd::Zero(FS.n_nodes_);
    Eigen::VectorXd init_Y = Eigen::VectorXd::Zero(FS.n_nodes_);
    Eigen::VectorXd init_T = Eigen::VectorXd::Zero(FS.n_nodes_tension_);
    init_Y[init_Y.size() - 1 - 3] = mlength / 2.0;
    init_Y[init_Y.size() - 1 - 2] = 1.0;
    Eigen::VectorXd XX(init_X.size() + init_Y.size() + init_T.size());
    XX << init_X, init_Y, init_T;

    // Try Divide and Construct
    auto Div = FS.DivideAndConstruct(XX, mlength);

    autodiff::VectorXreal YCtrue{{0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    autodiff::VectorXreal YsCtrue{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    EXPECT_TRUE(YCtrue.isApprox(Div.YC_));
    EXPECT_TRUE(YsCtrue.isApprox(Div.YsC_));
}

// Test penalty external functional divide and construct
TEST(FiberChebysehvPenaltyAutodiff, external_divide_and_construct) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    FiberChebyshevPenaltyAutodiffExternal<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);

    // Now create our fiber in XYT
    Eigen::VectorXd init_X = Eigen::VectorXd::Zero(FS.n_nodes_);
    Eigen::VectorXd init_Y = Eigen::VectorXd::Zero(FS.n_nodes_);
    Eigen::VectorXd init_T = Eigen::VectorXd::Zero(FS.n_nodes_tension_);
    init_Y[init_Y.size() - 1 - 3] = mlength / 2.0;
    init_Y[init_Y.size() - 1 - 2] = 1.0;
    Eigen::VectorXd XX(init_X.size() + init_Y.size() + init_T.size());
    XX << init_X, init_Y, init_T;

    // Try Divide and Construct
    auto Div = skelly_fiber::DivideAndConstructFCPA<autodiff::VectorXreal>(
        XX, mlength, FS.n_nodes_, FS.n_nodes_tension_, FS.n_equations_, FS.n_equations_tension_, FS.IM_, FS.IMT_);

    autodiff::VectorXreal YCtrue{{0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    autodiff::VectorXreal YsCtrue{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    EXPECT_TRUE(YCtrue.isApprox(Div.YC_));
    EXPECT_TRUE(YsCtrue.isApprox(Div.YsC_));
}

// Test the penalty forces
TEST(FiberChebysehvPenaltyAutodiff, real_forces) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);

    // Taken from Julia for the first timestep of the function in a sheer flow to compare to
    autodiff::VectorXreal XX{{60.0480227998087770,   -101.9506826907408765, 51.9343787940473760,   -10.4540573251306164,
                              -11.7091066585494996,  16.2140161411610997,   -11.7083971699872880,  6.0289382372732394,
                              -2.4070215350084792,   0.6838614991858294,    -0.0049060254621093,   -0.1788380697556702,
                              0.1602479731360698,    -0.0931454880346660,   0.0424927898922256,    -0.0128601494773640,
                              0.2193973350388452,    0.4087291888305368,    0.4841441405005029,    -1.5750008132303601,
                              52.8657357932827736,   -102.2105217746640591, 92.6318267662486647,   -69.9190532376784688,
                              39.8751997353422922,   -16.1017047444886963,  3.2404279967474698,    1.5672021673638288,
                              -2.3049734804756610,   1.6231572964223722,    -0.8211937355941242,   0.3092362423873997,
                              -0.0745206052414949,   -0.0036665860741279,   0.0277860139814897,    -0.0176334813873533,
                              0.4561181687022006,    0.9214887152110773,    -0.0592689415721022,   -0.2571209346937166,
                              -874.5796455234896030, 1281.7605985693051025, -953.9504954075138130, 554.5044090342805703,
                              -210.8615366227862467, -5.9283405118789814,   94.4457731090606671,   -97.9709166148476669,
                              66.5837307017072249,   -33.6612389195228161,  12.2160734371203681,   -2.1248214329907982,
                              -1.1479555519597531,   1.5078442418778060,    -1.0374971493440193,   0.5394800388411124,
                              52.0019688053351885,   -29.9310303532931457}};
    autodiff::VectorXreal oldXX{
        {14.7463835307163826, -20.6417455684296982, -0.4814512200205823, 15.2029164004962176, -13.0711212011266600,
         5.6033741423131938,  -1.1745630201570518,  -0.0218081216966133, 0.0902497715598546,  -0.0288303077441598,
         0.0044113201835056,  -0.0001266765326215,  -0.0001003834834743, 0.0000258121148767,  -0.0000030338682357,
         0.0000000832824501,  0.1213134041173209,   0.2237146559560383,  0.2609178824818563,  -0.7097415557391460,
         -0.0000000000000058, 0.0000000000000155,   0.0000000000000049,  -0.0000000000000044, 0.0000000000000134,
         -0.0000000000000011, -0.0000000000000016,  0.0000000000000036,  0.0000000000000015,  -0.0000000000000011,
         0.0000000000000005,  -0.0000000000000011,  -0.0000000000000010, 0.0000000000000012,  0.0000000000000002,
         -0.0000000000000005, 0.5000000000000000,   1.0000000000000000,  -0.0000000000000000, 0.0000000000000003,
         0.0000000000000143,  0.0000000000003712,   0.0000000000000013,  0.0000000000001477,  0.0000000000000488,
         -0.0000000000000233, 0.0000000000000439,   0.0000000000000318,  -0.0000000000000579, -0.0000000000000182,
         -0.0000000000000174, -0.0000000000000363,  0.0000000000000318,  0.0000000000000120,  -0.0000000000000330,
         0.0000000000000127,  0.0000000000000201,   -0.0000000000000323}};

    // Create the Div and oDiv structures for the forces
    auto Div = FS.DivideAndConstruct(XX, mlength);
    auto oDiv = FS.DivideAndConstruct(oldXX, mlength);

    // Check that Div and oDiv were constructed properly
    autodiff::VectorXreal Div_XC_true{{0.2193973350388452, 0.2372251964598805, 0.0028417909692916, -0.0072575080268174,
                                       0.0049220682333773, -0.0019806415920415, 0.0006288075956914, -0.0001750150807689,
                                       0.0000305396679530, 0.0000020331302382, -0.0000038444786937, 0.0000017418982979,
                                       -0.0000006196113489, 0.0000002143828926, -0.0000000569414079,
                                       -0.0000000006979729}};
    autodiff::VectorXreal Div_YC_true{{0.4561181687022006, 0.4515625963068547, 0.0001020867602687, 0.0026684088836761,
                                       -0.0018531603341006, 0.0004543453149293, 0.0001019565868943, -0.0001282533679407,
                                       0.0000606097502972, -0.0000208524497103, 0.0000057801717048, -0.0000011094038048,
                                       -0.0000000278733446, 0.0000001455281717, -0.0000000952669703,
                                       0.0000000376847738}};
    autodiff::VectorXreal Div_TC_true{{52.0019688053351885, -37.6922710996160930, -17.1097648763923829,
                                       4.6566647371035907, -2.9161696507584343, 1.5595969754505641, -0.6775182679046713,
                                       0.2087423130572356, -0.0160981174585723, -0.0339240830782111, 0.0301621192556019,
                                       -0.0161984068697774, 0.0063719194087564, -0.0019731985562813,
                                       -0.0000379321437554, 0.0004487631672255}};

    EXPECT_TRUE(Div.XC_.isApprox(Div_XC_true));
    EXPECT_TRUE(Div.YC_.isApprox(Div_YC_true));
    EXPECT_TRUE(Div.TC_.isApprox(Div_TC_true));

    // Try to construct forces...
    auto [FxC, FyC, AFxC, AFyC] = skelly_fiber::FiberForces(Div, oDiv, 1.0, FS.n_equations_);

    autodiff::VectorXreal FxC_true{{-40.1841203785645718, 0.5017825766590462, 9.0712863694151622, -16.5145450775445219,
                                    9.5953464141813125, -3.6092715878098778, 1.4098766726509195, -0.4952018482858689,
                                    0.0657161709844420, 0.0336277649985792, -0.0253961520203569, 0.0106105495088178,
                                    -0.0040614511866837, 0.0015090770703443, -0.0003403408446132, -0.0000742960021206}};
    autodiff::VectorXreal FyC_true{{-82.7967661465759335, -96.5916771352022039, -1.7248030743707261, 7.9949733389512332,
                                    -4.8481528887073466, 0.8363392578963538, 0.5946793408762425, -0.5721292242440633,
                                    0.2952960524967616, -0.1129445946283010, 0.0327832984308377, -0.0055083108173924,
                                    -0.0011599296515784, 0.0015423860238323, -0.0008602239479789, 0.0003418622315936}};
    autodiff::VectorXreal AFxC_true{{-63.7824450161000698, -23.3035560357534166, 10.7836258968863987,
                                     -16.4955245904501027, 8.4703120664858762, -2.7735375407740817, 1.2387107757277087,
                                     -0.5936150768277532, 0.1517986930820068, 0.0001546629404403, -0.0167467662919035,
                                     0.0079503204822146, -0.0025509555127807, 0.0007378155286263, -0.0001355325156950,
                                     -0.0000403933705770}};
    autodiff::VectorXreal AFyC_true{{-175.5273251911966952, -193.7496147725789228, 0.4083470410761354,
                                     10.6736355347067597, -7.4126413364017729, 1.8173812597189283, 0.4078263475770978,
                                     -0.5130134717624768, 0.2424390011916278, -0.0834097988372393, 0.0231206868216767,
                                     -0.0044376152154424, -0.0001114933744596, 0.0005821126922170, -0.0003810678771097,
                                     0.0001507390982731}};

    EXPECT_TRUE(FxC.isApprox(FxC_true));
    EXPECT_TRUE(FyC.isApprox(FyC_true));
    EXPECT_TRUE(AFxC.isApprox(AFxC_true));
    EXPECT_TRUE(AFyC.isApprox(AFyC_true));
}

// Test the deflection objective from Julia
TEST(FiberChebysehvPenaltyAutodiff, real_evolution_xy) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    double zeta = 1000.0;
    double dt = 1.0 / zeta / 4.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);

    // Taken from Julia for the first timestep of the function in a sheer flow to compare to
    autodiff::VectorXreal XX = Julia_SecondStep_XX;
    autodiff::VectorXreal oldXX = Julia_SecondStep_XX;

    // Create the Div and oDiv structures for the forces
    auto Div = FS.DivideAndConstruct(XX, mlength);
    auto oDiv = FS.DivideAndConstruct(oldXX, mlength);

    // Try to construct forces...
    auto [FxC, FyC, AFxC, AFyC] = skelly_fiber::FiberForces(Div, oDiv, 1.0, FS.n_equations_);

    // Try to construct the flow
    autodiff::VectorXreal UC = zeta * Div.YC_;
    autodiff::VectorXreal VC = autodiff::VectorXreal::Zero(Div.YC_.size());

    // Get the evolution equations
    auto [eqXC, eqYC] = skelly_fiber::FiberEvolution(AFxC, AFyC, Div, oDiv, UC, VC, dt);

    // XXX Currently this fails, so write out the equations, so we can compare to what we expected from Julia
    Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");

    autodiff::VectorXreal eqXC_true{
        {-0.12110045935008376116748252115940, -0.13041696931395618808124936549575, -0.00024011741515490728302723022480,
         0.00413358436807732126938574879205, -0.00347666125896568855083157423280, 0.00139757786498634849724209683330,
         -0.00019089394307293504208025702873, -0.00009068601455517021211420042315, 0.00005429398397426668011495759503,
         -0.00000354593989260237349455773086, -0.00001201911983736027329371532507, 0.00000943735837424655820261740896,
         -0.00000409246711197989346644240255, 0.00000098639743942784387821249691, 0.00000004865207879443243569507894,
         -0.00000017124479775025440439041125}};
    autodiff::VectorXreal eqYC_true{
        {0.00083629638311566445452172047226, -0.00107061693862159357237828771758, -0.00031138160225468774957441331352,
         0.00116758273133083672647158923752, -0.00083258792334591380946556826714, 0.00015962221810961754958971270391,
         0.00019455921936148808581643065985, -0.00021335438417505181440958494932, 0.00011646354556131292205638227966,
         -0.00003890865084281206577121806078, 0.00000546705748870515882632988014, 0.00000214165716515491101119188569,
         -0.00000169446253619582499942242213, 0.00000057113173931634273286663374, -0.00000009882577524035701261197179,
         -0.00000000356083196810410894713202}};

    for (auto i = 0; i < eqXC.size(); i++) {
        EXPECT_NEAR(eqXC[i].val(), eqXC_true[i].val(), 1e-10);
        EXPECT_NEAR(eqYC[i].val(), eqYC_true[i].val(), 1e-10);
    }
}

// Test the tension equation
TEST(FiberChebysehvPenaltyAutodiff, real_tension) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    double zeta = 1000.0;
    double dt = 1.0 / zeta / 4.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);

    // Taken from Julia for the first timestep of the function in a sheer flow to compare to
    autodiff::VectorXreal XX = Julia_SecondStep_XX;
    autodiff::VectorXreal oldXX = Julia_SecondStep_XX;

    // Create the Div and oDiv structures for the forces
    auto Div = FS.DivideAndConstruct(XX, mlength);
    auto oDiv = FS.DivideAndConstruct(oldXX, mlength);

    // Try to construct the flow
    autodiff::VectorXreal UsC = zeta * Div.YsC_;
    autodiff::VectorXreal VsC = autodiff::VectorXreal::Zero(Div.YsC_.size());
    autodiff::VectorXreal oUsC = zeta * oDiv.YC_;
    autodiff::VectorXreal oVsC = autodiff::VectorXreal::Zero(oDiv.YsC_.size());

    autodiff::VectorXreal eqTC =
        skelly_fiber::FiberPenaltyTension(Div, oDiv, UsC, VsC, oUsC, oVsC, dt, FS.n_equations_tension_);

    autodiff::VectorXreal eqTC_true{{891.92151983293001649144571274518967, -654.54080906886895263596670702099800,
                                     339.62812768393706619463046081364155, -63.10414928298077086310513550415635,
                                     -93.18622792579498081977362744510174, 145.30077225011501695917104370892048,
                                     -123.79649573338170398528745863586664, 69.18368540061042892830300843343139,
                                     -23.01496523415610084839499904774129, 1.26216785094897265828706167667406,
                                     3.37585681166985240864164552476723, -2.15781791540036849141870334278792,
                                     0.72220404775986279943822410132270, -0.12418267072039373966063635634782,
                                     -0.00842310534677920969004460971519, 0.01261420272924905668088246812886}};

    for (auto i = 0; i < eqTC.size(); i++) {
        EXPECT_NEAR(eqTC[i].val(), eqTC_true[i].val(), 1e-10);
    }
}

// Test the clamped boundary conditions
TEST(FiberChebysehvPenaltyAutodiff, real_clampedbc) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);

    // Taken from Julia for the first timestep of the function in a sheer flow to compare to
    autodiff::VectorXreal XX = Julia_SecondStep_XX;
    autodiff::VectorXreal oldXX = Julia_SecondStep_XX;

    // Create the Div and oDiv structures for the forces
    auto Div = FS.DivideAndConstruct(XX, mlength);
    auto oDiv = FS.DivideAndConstruct(oldXX, mlength);

    // Get the clamped boundary conditions
    autodiff::VectorXreal cposition{{0.0, 0.0}};
    autodiff::VectorXreal cdirector{{0.0, 1.0}};
    autodiff::VectorXreal X1;
    autodiff::VectorXreal X2;
    autodiff::VectorXreal Y1;
    autodiff::VectorXreal Y2;
    autodiff::VectorXreal T;
    FiberBoundaryCondition<autodiff::real> BCL = skelly_fiber::ClampedBC<autodiff::real, autodiff::VectorXreal>(
        Div, oDiv, skelly_fiber::FSIDE::left, cposition, cdirector);

    EXPECT_NEAR(BCL.T_.val(), -133.35780960147738, 1e-10);
}

// ********************************************************************************************************************
// Physics tests
// ********************************************************************************************************************

// Test a full sheer flow implementation jacobian (initial step)
TEST(FiberChebysehvPenaltyAutodiff, real_jacobian_initial_step) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    double zeta = 1000.0;
    double dt = 1.0 / zeta / 4.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);

    // Now create our fiber in XYT
    autodiff::VectorXreal init_X = Eigen::VectorXd::Zero(FS.n_nodes_);
    autodiff::VectorXreal init_Y = Eigen::VectorXd::Zero(FS.n_nodes_);
    autodiff::VectorXreal init_T = Eigen::VectorXd::Zero(FS.n_nodes_tension_);
    init_Y[init_Y.size() - 1 - 3] = mlength / 2.0;
    init_Y[init_Y.size() - 1 - 2] = 1.0;
    autodiff::VectorXreal XX(init_X.size() + init_Y.size() + init_T.size());
    XX << init_X, init_Y, init_T;
    autodiff::VectorXreal oldXX = XX;

    // Create the evaluation vector
    autodiff::VectorXreal F;

    // Try to push a jacobian through this
    Eigen::MatrixXd J = autodiff::jacobian(SheerDeflectionObjective<autodiff::VectorXreal>, autodiff::wrt(XX),
                                           autodiff::at(XX, FS, oldXX, mlength, zeta, dt), F);

    // Test against outputs from Julia where available above the double precision problem
    // Test the function evaluated at XX
    EXPECT_DOUBLE_EQ(-0.125, F[0].val());
    EXPECT_DOUBLE_EQ(-0.125, F[1].val());

    // Test the jacobian at specific test points that aren't close to 0 to avoid double precision issues
    for (auto i = 0; i < 1; ++i) {
        for (auto j = 0; j < 1; ++j) {
            EXPECT_NEAR(J(i, j), Julia_FirstStep_Jacobian(i, j), 1e-10);
        }
    }
}

// Test a full sheer flow implementation jacobian (second step)
TEST(FiberChebysehvPenaltyAutodiff, real_jacobian_next_step) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    double zeta = 1000.0;
    double dt = 1.0 / zeta / 4.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);

    autodiff::VectorXreal XX = Julia_SecondStep_XX;
    autodiff::VectorXreal oldXX = XX;

    // Create the evaluation vector
    autodiff::VectorXreal F;

    // Try to push a jacobian through this
    Eigen::MatrixXd J = autodiff::jacobian(SheerDeflectionObjective<autodiff::VectorXreal>, autodiff::wrt(XX),
                                           autodiff::at(XX, FS, oldXX, mlength, zeta, dt), F);

    // Test against outputs from Julia where available above the double precision problem
    // Test the function evaluated at XX
    EXPECT_NEAR(-0.12110045935008376, F[0].val(), 1e-10);
    EXPECT_NEAR(-0.1304169693139562, F[1].val(), 1e-10);

    // Test the jacobian at specific test points that aren't close to 0 to avoid double precision issues
    for (auto i = 0; i < J.rows(); ++i) {
        for (auto j = 0; j < J.cols(); ++j) {
            EXPECT_NEAR(J(i, j), Julia_SecondStep_Jacobian(i, j), 1e-10);
        }
    }
}
