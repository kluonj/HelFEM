#ifndef RDMFT_TEST_HELPERS_H
#define RDMFT_TEST_HELPERS_H

#include "../../atomic/TwoDBasis.h"
#include "../../atomic/basis.h"
#include "../../general/model_potential.h"
#include <memory>

namespace helfem { namespace rdmft { namespace test {
    inline helfem::atomic::basis::TwoDBasis make_test_basis(){
    using helfem::atomic::basis::TwoDBasis;
    using helfem::modelpotential::POINT_NUCLEUS;
    auto poly = std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(4,15));
    arma::ivec lval; arma::ivec mval; helfem::atomic::basis::angular_basis(0,0,lval,mval);
    arma::vec bval = helfem::atomic::basis::form_grid(helfem::modelpotential::POINT_NUCLEUS, 0.0, 10, 20.0, 4, 2.0, 0, 4, 2.0, 0, 0, 0, 0.0);
    TwoDBasis basis(1, POINT_NUCLEUS, 1.0, poly, false, 0, bval, 0, lval, mval, 0, 0, 0.0);
    return basis;
}
}}}

#endif
