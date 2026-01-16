#include <iostream>
#include "../general/checkpoint.h"
#include "../atomic/TwoDBasis.h"
#include "../rdmft/rdmft_energy.h"

using namespace helfem;

int main(int argc, char** argv) {
  std::string chkf = "helfem_hf.chk";
  if(argc>1) chkf = argv[1];
  try {
    Checkpoint chk(chkf, false);
    atomic::basis::TwoDBasis basis; chk.read(basis);

    arma::mat Ca; chk.read("Ca", Ca);
    arma::mat Cb; if(chk.exist("Cb")) chk.read("Cb", Cb);
    arma::mat H0; chk.read("H0", H0);

    int nela=0, nelb=0;
    if(chk.exist("nela")) chk.read("nela", nela);
    if(chk.exist("nelb")) chk.read("nelb", nelb);

    arma::mat Caocc, Cbocc;
    if(nela>0) Caocc = Ca.cols(0, std::min<arma::uword>(Ca.n_cols, (arma::uword)nela)-1);
    if(nelb>0 && Cb.n_elem) Cbocc = Cb.cols(0, std::min<arma::uword>(Cb.n_cols, (arma::uword)nelb)-1);
    else if(nelb>0 && !Cb.n_elem) Cbocc = Caocc;

    arma::uword Norb = Caocc.n_cols + Cbocc.n_cols;
    arma::mat C_AO;
    if(Norb>0) {
      C_AO.set_size(Ca.n_rows, Norb);
      for(arma::uword i=0;i<Caocc.n_cols;i++) C_AO.col(i)=Caocc.col(i);
      for(arma::uword j=0;j<Cbocc.n_cols;j++) C_AO.col(Caocc.n_cols+j)=Cbocc.col(j);
    }

    arma::vec n; if(Norb>0) { n.ones(Norb); }

    double Ecore = helfem::rdmft::core_energy<atomic::basis::TwoDBasis>(H0, C_AO, n);
    double EJ = helfem::rdmft::hartree_energy<atomic::basis::TwoDBasis>(basis, C_AO, n);
    double EXC = helfem::rdmft::muller_xc_energy<atomic::basis::TwoDBasis>(basis, C_AO, n, 1.0);

    double Enucr=0.0; if(chk.exist("Enucr")) chk.read("Enucr", Enucr);
    double Eef=0.0; if(chk.exist("Eefield")) chk.read("Eefield", Eef);
    double Emf=0.0; if(chk.exist("Emfield")) chk.read("Emfield", Emf);
    double Econf=0.0; if(chk.exist("Econf")) chk.read("Econf", Econf);

    double Etot_r = Ecore + EJ + EXC + Enucr + Eef + Emf + Econf;

    std::cout.setf(std::ios::fixed); std::cout.precision(12);
    std::cout << "RDMFT (p=1.0) from checkpoint: \n";
    std::cout << "  E_core = " << Ecore << "\n";
    std::cout << "  E_J    = " << EJ << "\n";
    std::cout << "  E_xc   = " << EXC << "\n";
    std::cout << "  Enucr  = " << Enucr << "\n";
    std::cout << "  Etot_r = " << Etot_r << "\n";

    if(chk.exist("Etot")) {
      double Etot; chk.read("Etot", Etot);
      std::cout << "Checkpoint Etot = " << Etot << "\n";
      std::cout << "Difference = " << std::abs(Etot_r - Etot) << "\n";
    }
    return 0;
  } catch(std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 2;
  }
}
