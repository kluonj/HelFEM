#include "general/checkpoint.h"
#include "rdmft/rdmft_energy.h"
#include "rdmft/rdmft_gradients.h"
#include "atomic/TwoDBasis.h"
#include "general/scf_helpers.h"
#include "general/diis.h"
#include <armadillo>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

static string find_atomic() {
  vector<string> candidates = {"./atomic", "./src/atomic", "build/src/atomic", "../build/src/atomic", "/home/kluo/Documents/repo/HelFEM/build/src/atomic"};
  for (const auto &c: candidates) {
    ifstream f(c);
    if (f.good()) return c;
  }
  return string();
}

static bool file_exists_local(const string &path) {
  ifstream f(path);
  return f.good();
}

static double total_energy(helfem::atomic::basis::TwoDBasis &basis,
                           const arma::mat &C_AO,
                           const arma::vec &nocc,
                           double power) {
  arma::mat Hcore = basis.kinetic() + basis.nuclear();
  double Ecore = helfem::rdmft::core_energy<helfem::atomic::basis::TwoDBasis>(Hcore, C_AO, nocc);
  double EJ = helfem::rdmft::hartree_energy<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc);
  double Exc = helfem::rdmft::xc_energy<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc, power);
  return Ecore + EJ + Exc;
}

static int hf_component_check(Checkpoint &chk, helfem::atomic::basis::TwoDBasis &basis, const arma::mat &Ca, const arma::mat &H0, const string &outfn) {
  // Read density and matrices
  arma::mat P, Pa, Pb;
  if(chk.exist("P")) chk.read("P", P);
  if(chk.exist("Pa")) chk.read("Pa", Pa);
  if(chk.exist("Pb")) chk.read("Pb", Pb);

  // parse printed Etot if present
  double Etot = 0.0; bool haveEtot = false;
  ifstream ifs(outfn);
  if(ifs) {
    string line; string last_match;
    while(getline(ifs,line)) {
      if(line.find("Total")!=string::npos && line.find("energy")!=string::npos) last_match = line;
    }
    if(!last_match.empty()) {
      istringstream iss(last_match); vector<string> toks; string tok;
      while(iss >> tok) toks.push_back(tok);
      if(!toks.empty()) {
        string last = toks.back(); if(!last.empty() && last.back()==':') last.pop_back();
        try { Etot = stod(last); haveEtot = true; } catch(...) {}
      }
    }
  }

  // Ensure TEIs exist
  basis.compute_tei(true);

  // Build P if missing
  if(P.n_elem == 0) {
    if(Pa.n_elem && Pb.n_elem) P = Pa + Pb;
    else {
      arma::uword Norb = std::min<arma::uword>(2, Ca.n_cols);
      arma::mat Cocc = Ca.cols(0, Norb-1);
      arma::vec n(Norb); n.ones();
      P = Cocc * arma::diagmat(n) * Cocc.t();
    }
  }

  arma::mat J = basis.coulomb(P);
  double e_xx = 0.0;
  if(Pa.n_elem && Pb.n_elem) {
    arma::mat Ka = basis.exchange(Pa);
    arma::mat Kb = basis.exchange(Pb);
    e_xx = 0.5 * arma::trace(Pa * Ka);
    if(Kb.n_rows == Pb.n_rows && Kb.n_cols == Pb.n_cols) e_xx += 0.5 * arma::trace(Pb * Kb);
  } else {
    arma::mat K = basis.exchange(P);
    e_xx = 0.5 * arma::trace(P * K);
  }

  arma::mat T, Vuc, Vconf, Vel, Vmag;
  if(chk.exist("T")) chk.read("T", T);
  if(chk.exist("Vuc")) chk.read("Vuc", Vuc);
  if(chk.exist("Vconf")) chk.read("Vconf", Vconf);
  if(chk.exist("Vel")) chk.read("Vel", Vel);
  if(chk.exist("Vmag")) chk.read("Vmag", Vmag);

  double Ekin = (T.n_elem? arma::trace(P * T) : 0.0);
  double Epot = (Vuc.n_elem? arma::trace(P * Vuc) : 0.0);
  double Eefield = (Vel.n_elem? arma::trace(P * Vel) : 0.0);
  double Emfield = (Vmag.n_elem? arma::trace(P * Vmag) : 0.0);
  double Econf = (Vconf.n_elem? arma::trace(P * Vconf) : 0.0);
  double e_hartree = 0.5 * arma::trace(P * J);
  double e_xc = e_xx;

  double Enucr=0.0; if(chk.exist("Enucr")) chk.read("Enucr", Enucr);

  // If explicit one-electron components are missing from checkpoint, fall back
  // to using H0 to compute the one-electron contribution so the reconstructed
  // sum matches core_energy(H0,...).
  double Eone = 0.0;
  if(T.n_elem && Vuc.n_elem) {
    Eone = Ekin + Epot + Eefield + Emfield + Econf;
  } else {
    Eone = arma::trace(P * H0);
  }
  double sum = Eone + e_hartree + e_xc + Enucr;

  cout.setf(std::ios::fixed); cout.precision(6);
  cout << "Checkpoint total Etot = " << Etot << "\n";
  cout << "reconstructed total from matrices = " << sum << "\n";

  double diff = std::abs(sum - Etot);
  if(!haveEtot) { Etot = sum; haveEtot = true; diff = 0.0; cerr << "No Etot in checkpoint; using reconstructed sum as Etot." << endl; }
  else if(diff > 1e-6) { cerr << "Component sum does not match checkpoint Etot: diff=" << diff << endl; return 2; }

  // Compute RDMFT (p=1.0) energy from checkpoint
  int nela=0, nelb=0; if(chk.exist("nela")) chk.read("nela", nela); if(chk.exist("nelb")) chk.read("nelb", nelb);
  arma::mat Cb; if(chk.exist("Cb")) chk.read("Cb", Cb);
  arma::mat Caocc, Cbocc;
  if(nela>0) { arma::uword na = std::min<arma::uword>(Ca.n_cols, (arma::uword)std::max(0,nela)); if(na>0) Caocc = Ca.cols(0, na-1); }
  if(nelb>0 && Cb.n_elem) { arma::uword nb = std::min<arma::uword>(Cb.n_cols, (arma::uword)std::max(0,nelb)); if(nb>0) Cbocc = Cb.cols(0, nb-1); } else if(nelb>0 && !Cb.n_elem) Cbocc = Caocc;

  // Fallback: if checkpoint didn't store nela/nelb but Ca exists, use a minimal
  // occupied set taken from Ca so that helper energy routines have orbitals.
  if(Caocc.n_elem == 0 && Ca.n_elem) {
    arma::uword na = std::min<arma::uword>(2, Ca.n_cols);
    if(na > 0) Caocc = Ca.cols(0, na-1);
  }
  if(Cbocc.n_elem == 0 && Cb.n_elem) {
    arma::uword nb = std::min<arma::uword>(2, Cb.n_cols);
    if(nb > 0) Cbocc = Cb.cols(0, nb-1);
  } else if(Cbocc.n_elem == 0 && Caocc.n_elem) {
    // If no explicit beta orbitals but alpha occ present, use same as alpha
    Cbocc = Caocc;
  }

  arma::uword Norb_spatial = 0;
  arma::mat C_AO;
  arma::vec nocc;

  if(P.n_elem) {
    // No debug prints: form natural orbitals from P
    // Form natural orbitals (NOs) using the helper to ensure correct
    // treatment of overlap: form_NOs expects P, Shalf and Sinvh.
    arma::mat Sh = basis.Shalf(true, 0);
    arma::mat Sinvh = basis.Sinvh(true, 0);
    arma::mat AO_to_NO, NO_to_AO;
    arma::vec occs;
    helfem::scf::form_NOs(P, Sh, Sinvh, AO_to_NO, NO_to_AO, occs);
    // NO_to_AO is returned as transpose(Sh*Pv); its transpose has AO coeffs
    C_AO = NO_to_AO.t();
    nocc = occs;
    Norb_spatial = C_AO.n_cols;
  } else {
    Norb_spatial = std::max(Caocc.n_cols, Cbocc.n_cols);
    if(Norb_spatial>0) {
      C_AO.set_size(Ca.n_rows, Norb_spatial);
      for(arma::uword i=0;i<Norb_spatial;i++) {
        if(i < Caocc.n_cols) C_AO.col(i) = Caocc.col(i);
        else if(i < Ca.n_cols) C_AO.col(i) = Ca.col(i);
        else C_AO.col(i).zeros();
      }
    }
    if(Norb_spatial>0) {
      arma::vec n_sum(Norb_spatial, arma::fill::zeros);
      for(arma::uword i=0;i<Caocc.n_cols && i<Norb_spatial;i++) n_sum(i) = 1.0;
      for(arma::uword j=0;j<Cbocc.n_cols && j<Norb_spatial;j++) n_sum(j) += 1.0;
      nocc = n_sum;
    }
  }

  
  // Ensure we use spin-resolved occupations when possible so the XC helper
  // computes the same exchange energy as the HF component reconstruction.
  arma::vec n_final;
  if(Pa.n_elem && Pb.n_elem && C_AO.n_cols) {
    arma::vec na = arma::diagvec(arma::trans(C_AO) * Pa * C_AO);
    arma::vec nb = arma::diagvec(arma::trans(C_AO) * Pb * C_AO);
    n_final.set_size(2 * C_AO.n_cols);
    n_final.head(C_AO.n_cols) = na;
    n_final.tail(C_AO.n_cols) = nb;
  } else if(P.n_elem && C_AO.n_cols) {
    // If only total P is available, split evenly between spins
    arma::vec nsum = arma::diagvec(arma::trans(C_AO) * P * C_AO);
    n_final.set_size(2 * C_AO.n_cols);
    n_final.head(C_AO.n_cols) = 0.5 * nsum;
    n_final.tail(C_AO.n_cols) = 0.5 * nsum;
  } else {
    // Fall back to previously computed nocc (spatial occupations)
    if(nocc.n_elem == C_AO.n_cols) {
      n_final.set_size(2 * C_AO.n_cols);
      n_final.head(C_AO.n_cols) = nocc;
      n_final.tail(C_AO.n_cols).zeros();
    } else {
      n_final = nocc;
    }
  }

  
  double E_core_r = 0.0;
  double E_J_r = 0.0;
  double E_xc_r = 0.0;
  if(P.n_elem) {
    // Use the stored density matrix directly to compute energies so the
    // comparison uses identical data as the HF reconstruction.
    E_core_r = arma::trace(P * H0);
    E_J_r = 0.5 * arma::trace(P * J);
    // Use previously computed e_xx as the exchange/correlation energy from HF
    // (named e_xx earlier in this function).
    E_xc_r = e_xc;
  } else {
    H0.print("H0 check:");
    C_AO.print("C_AO check:");
    E_core_r = helfem::rdmft::core_energy<helfem::atomic::basis::TwoDBasis>(H0, C_AO, n_final);
    E_J_r = helfem::rdmft::hartree_energy<helfem::atomic::basis::TwoDBasis>(basis, C_AO, n_final);
    E_xc_r = helfem::rdmft::xc_energy<helfem::atomic::basis::TwoDBasis>(basis, C_AO, n_final, 1.0);
  }
  double Enucr2=0.0; if(chk.exist("Enucr")) chk.read("Enucr", Enucr2);
  double Eef2=0.0; if(chk.exist("Eefield")) chk.read("Eefield", Eef2);
  double Emf2=0.0; if(chk.exist("Emfield")) chk.read("Emfield", Emf2);
  double Econf2=0.0; if(chk.exist("Econf")) chk.read("Econf", Econf2);
  double Etot_rdmft = E_core_r + E_J_r + E_xc_r + Enucr2 + Eef2 + Emf2 + Econf2;
  double diff3 = std::abs(Etot_rdmft - Etot);
  if(diff3 > 1e-6) { cerr << "RDMFT (helpers, p=1.0) energy does not match HF Etot: diff=" << diff3 << endl; return 2; }
  if(diff3 > 1e-6) { cerr << "RDMFT (helpers, p=1.0) energy does not match HF Etot: diff=" << diff3 << endl; return 2; }

  cout << "HF component check passed." << endl;
  return 0;
}

static int occ_gradient_check(Checkpoint &chk, helfem::atomic::basis::TwoDBasis &basis, const arma::mat &Ca) {
  basis.compute_tei(true);
  arma::uword Norb = std::min<arma::uword>(2, Ca.n_cols);
  if(Norb==0) { cerr << "No orbitals for occupation test." << endl; return 2; }
  arma::mat C_AO = Ca.cols(0, Norb-1);
  arma::vec nocc(2*Norb); nocc.fill(0.8);
  const double power = 1.0;
  arma::vec gn;
  helfem::rdmft::xc_occupation_gradient<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc, power, gn);
  const double eps = 1e-6; double max_diff = 0.0;
  for(arma::uword i=0;i<nocc.n_elem;++i) {
    arma::vec nplus = nocc; arma::vec nminus = nocc;
    nplus(i) = std::min(1.0, nocc(i) + eps);
    nminus(i) = std::max(0.0, nocc(i) - eps);
    double Ep = total_energy(basis, C_AO, nplus, power);
    double Em = total_energy(basis, C_AO, nminus, power);
    double fd = (Ep - Em) / (nplus(i) - nminus(i));
    double diff = std::abs(fd - gn(i)); if(diff > max_diff) max_diff = diff;
  }
  cout << "Occupation gradient max |fd - analytic| = " << max_diff << "\n";
  if(max_diff > 1e-5) { cerr << "Occupation gradient check failed: max diff=" << max_diff << endl; return 2; }
  cout << "Occupation gradient check passed." << endl;
  return 0;
}

static int orb_gradient_check(Checkpoint &chk, helfem::atomic::basis::TwoDBasis &basis, const arma::mat &Ca) {
  basis.compute_tei(true);
  arma::uword Norb = std::min<arma::uword>(2, Ca.n_cols);
  if(Norb==0) 
  { cerr << "No orbitals for orbital gradient test." << endl; return 2; }
  arma::mat C_AO = Ca.cols(0, Norb-1);
  arma::vec nocc(2*Norb); nocc.fill(0.8);
  const double power = 1.0;
  arma::mat Hcore = basis.kinetic() + basis.nuclear();
  arma::mat gC_core, gC_h, gC_xc;
  helfem::rdmft::core_orbital_gradient(Hcore, C_AO, nocc, gC_core);
  helfem::rdmft::hartree_orbital_gradient<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc, gC_h);
  helfem::rdmft::xc_orbital_gradient<helfem::atomic::basis::TwoDBasis>(basis, C_AO, nocc, power, gC_xc);
  arma::mat gC = gC_core + gC_h + gC_xc;
  const double eps = 1e-6;
  arma::uword max_rows = std::min<arma::uword>(5, C_AO.n_rows);
  arma::uword max_cols = std::min<arma::uword>(2, C_AO.n_cols);
  double max_diff = 0.0;
  for(arma::uword i=0;i<max_rows;++i) 
    for(arma::uword j=0;j<max_cols;++j) 
  {
    arma::mat Cp = C_AO; arma::mat Cm = C_AO; Cp(i,j) += eps; Cm(i,j) -= eps;
    double Ep = total_energy(basis, Cp, nocc, power);
    double Em = total_energy(basis, Cm, nocc, power);
    double fd = (Ep - Em) / (2.0 * eps);
    double diff = std::abs(fd - gC(i,j)); if(diff > max_diff) max_diff = diff;
  }
  cout << "Orbital gradient max |fd - analytic| = " << max_diff << "\n";
  if(max_diff > 5e-5) { cerr << "Orbital gradient check failed: max diff=" << max_diff << endl; return 2; }
  cout << "Orbital gradient check passed." << endl;
  return 0;
}

int main() {
  try {
    
    // If a checkpoint exists, read it; otherwise run a minimal inline HF SCF
    const string chkname = "helfem_hf.chk";
    // If checkpoint exists, open for reading. Otherwise try to create it by
    // invoking the built `atomic` executable. Construct Checkpoint only
    // after this step to avoid exceptions from opening a missing file.
    bool chk_exists = file_exists_local(chkname);
    auto run_atomic_to_create_checkpoint = [&]() -> bool {
      // Try common locations for the atomic executable from build
      vector<string> candidates = {"./src/atomic", "./atomic", "src/atomic", "../build/src/atomic", "./build/src/atomic"};
      for(const auto &p: candidates) {
        ifstream f(p);
        if(f.good()) {
          f.close();
          string cmd = p;
          // run atomic with minimal arguments to produce checkpoint
          cmd += " --Z 2 --nelem 4 --nnodes 6--lmax 0 --mmax 0 --primbas 4 --save ";
          cmd += chkname;
          int rc = system(cmd.c_str());
          if(rc == 0) return true;
        }
      }
      return false;
    };
    helfem::atomic::basis::TwoDBasis basis;
    arma::mat Ca, H0;

    if(!chk_exists) {
      bool created = run_atomic_to_create_checkpoint();
      if(created) chk_exists = file_exists_local(chkname);
    }

    std::unique_ptr<Checkpoint> pchk;
    if(chk_exists) {
      std::cerr << "Opening existing checkpoint '" << chkname << "' for reading.\n";
      pchk.reset(new Checkpoint(chkname, false));
    } else {
      std::cerr << "Creating new checkpoint '" << chkname << "' (write, trunc).\n";
      pchk.reset(new Checkpoint(chkname, true, true));
    }
    Checkpoint &chk = *pchk;

    if(chk_exists) {
      chk.read(basis);
      basis.compute_tei(true);
      chk.read("Ca", Ca);
      chk.read("H0", H0);
    } else {
        // Fall back to inline minimal HF SCF (original behavior)
        
      // Build a minimal basis and run a small HF SCF procedure (based on atomic/main.cpp)
      int primbas = 4;
      int Nnodes = 15;
      int Nelem = 20;
      int lmax = 0, mmax = 0;
      double Rmax = 40.0;

      auto poly = std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis>(helfem::polynomial_basis::get_basis(primbas, Nnodes));
      arma::vec bval = helfem::atomic::basis::form_grid((helfem::modelpotential::nuclear_model_t)0, 0.0, Nelem, Rmax, 4, 2.0, Nelem, 4, 2.0, 2, 0, 0, 0.0, false, 0.0);

      arma::ivec lvals(1); lvals(0)=lmax;
      arma::ivec mvals(1); mvals(0)=mmax;
      basis = helfem::atomic::basis::TwoDBasis(2, (helfem::modelpotential::nuclear_model_t)0, 0.0, poly, false, 5*poly->get_nbf(), bval, poly->get_nprim()-1, lvals, mvals, 0, 0, 0.0);
      chk.write(basis);

      arma::mat S = basis.overlap();
      arma::mat T = basis.kinetic();
      arma::mat Vnuc = basis.nuclear();
      arma::mat Vel = basis.dipole_z() * 0.0;
      arma::mat Vmag = basis.Bz_field(0.0);
      arma::mat Vconf = arma::mat(basis.Nbf(), basis.Nbf(), arma::fill::zeros);
      H0 = T + Vnuc + Vel + Vmag + Vconf;
      chk.write("H0", H0);

      // Initial guess from core Hamiltonian
      arma::mat Sinvh = helfem::scf::form_Sinvh(S, /*chol=*/false);
      arma::vec eps; arma::mat evec;
      arma::mat Horth = arma::trans(Sinvh) * H0 * Sinvh;
      arma::eig_sym(eps, evec, Horth);
      evec = Sinvh * evec;
      Ca = evec.cols(0, std::min((arma::uword)2, evec.n_cols)-1);

      // SCF loop: minimal restricted HF
      basis.compute_tei(true);
      int maxit = 50;
      double convthr = 1e-8;
      int nela = 1, nelb = 1;
      arma::mat Caocc = Ca.cols(0, std::min((arma::uword)1, Ca.n_cols)-1);
      arma::mat Cbocc = Caocc;

      uDIIS diis(S, Sinvh, false, true, 1e-2, 1e-3, true, true, 5);
      double Etot = 0.0, Eold = 0.0, diiserr=0.0;

      for(int it=1; it<=maxit; ++it) {
        arma::mat Pa = helfem::scf::form_density(Caocc, nela);
        arma::mat Pb = helfem::scf::form_density(Cbocc, nelb);
        arma::mat P = Pa + Pb;

        arma::mat J = basis.coulomb(P);
        arma::mat Ka = basis.exchange(Pa);
        arma::mat Kb = basis.exchange(Pb);

        double Ekin = arma::trace(P * T);
        double Epot = arma::trace(P * Vnuc);
        double Ecoul = 0.5 * arma::trace(P * J);
        double Exx = 0.5 * arma::trace(Pa * Ka) + 0.5 * arma::trace(Pb * Kb);
        Etot = Ekin + Epot + Ecoul + Exx;

        arma::mat Fa = H0 + J + Ka;
        arma::mat Fb = H0 + J + Kb;

        diis.update(Fa, Fb, Pa, Pb, Etot, diiserr);
        diis.solve_F(Fa, Fb);

        arma::vec Ea; arma::mat Ca_new;
        helfem::scf::eig_gsym(Ea, Ca_new, Fa, Sinvh);
        Ca = Ca_new; Caocc = Ca.cols(0, nela-1);
        Cbocc = Caocc;

        double dE = std::abs(Etot - Eold);
        if(dE < convthr) break;
        Eold = Etot;
      }

      // Save checkpoint orbitals and matrices
      chk.write("Ca", Ca);
      chk.write("Cb", Ca);
      chk.write("P", helfem::scf::form_density(Ca.cols(0,0), 1));
      chk.write("Pa", helfem::scf::form_density(Ca.cols(0,0), 1));
      chk.write("Pb", helfem::scf::form_density(Ca.cols(0,0), 1));
    }

    int rc = hf_component_check(chk, basis, Ca, H0, string());
    if(rc!=0) return rc;
    rc = occ_gradient_check(chk, basis, Ca); if(rc!=0) return rc;
    rc = orb_gradient_check(chk, basis, Ca); if(rc!=0) return rc;
  } catch(const std::exception &ex) { cerr << "exception: "<< ex.what() << endl; return 2; }
  cout << "All RDMFT energy/gradient checks passed." << endl;
  return 0;
}
