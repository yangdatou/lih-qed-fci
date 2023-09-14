import numpy, scipy

# Imports related to the quantum chemistry package PySCF
import pyscf
from pyscf import gto, fci
from pyscf.lo.orth import orth_ao

# Import related to electron-phonon interaction
import eph

def main(mol="lih", basis_set="sto3g", bond_lengths=None, alph=None):
    """
    Calculate energies of the H-Li system for a given bond length using FCI method, optionally including electron-phonon interaction.

    Parameters:
    - bond_length: distance between H and Li atoms.
    - basis_set: basis set to be used in the quantum calculation.
    - with_eph: a flag to indicate if electron-phonon interaction should be included.

    Returns:
    - List of energies for different states.
    """
    with_eph = (alph is not None)

    assert mol == "lih"
    nsinglet = 4
    ntriplet = 4
    nstate = (nsinglet + ntriplet) * 5

    out = open("%s-%s-%s.out" % (mol, basis_set, "fci" if alph is None else "%6.4f" % alph), "w")
    title  = ["# bond_length"]
    for istate in range(nsinglet):
        title.append("Energy(S%d)" % istate)
        title.append("N(S%d)" % istate)

    for istate in range(ntriplet):
        title.append("Energy(T%d)" % istate)
        title.append("N(T%d)" % istate)

    ci0 = None
    out.write("%12s, " * len(title) % tuple(title) + "\n")

    for bond_length in bond_lengths:
        # Define the molecular system
        m = pyscf.gto.Mole()
        m.verbose = 0
        m.atom = f'H 0 0 0; Li 0 0 {bond_length}'
        m.unit = 'B'
        m.basis = basis_set
        m.build()

        # Orthogonalize atomic orbitals using meta-Lowdin method
        coeff_lo = orth_ao(m, 'meta_lowdin')
        nao, norb = coeff_lo.shape

        singlet_data = []
        triplet_data = []

        if not with_eph:
            # Perform a regular FCI calculation without electron-phonon interaction
            fci_obj = fci.FCI(m, mo=coeff_lo, singlet=False)
            fci_obj.nroots = nstate
            fci_obj.max_cycle = 1000
            fci_obj.conv_tol  = 1e-6
            fci_obj.verbose   = 10
            e0, c0 = fci_obj.kernel(ci0=ci0)

            singlet_count = 0
            triplet_count = 0

            for istate in range(nstate):
                ss = fci.spin_op.spin_square(c0[istate], norb, m.nelec)[1]
                np = 0.0
                is_singlet = abs(ss - 1.0) < 1e-6
                is_triplet = abs(ss - 3.0) < 1e-6
                print("State %d: energy = %12.6f, S^2 = %6.4f, N = %6.4f" % (istate, e0[istate], ss, np))
                assert is_singlet or is_triplet

                if is_singlet and singlet_count < nsinglet:
                    singlet_count += 1
                    singlet_data.append(e0[istate])
                    singlet_data.append(np)

                if is_triplet and triplet_count < ntriplet:
                    triplet_count += 1
                    triplet_data.append(e0[istate])
                    triplet_data.append(np)

                if singlet_count == nsinglet and triplet_count == ntriplet:
                    break

        else:
            # Include electron-phonon interaction in FCI calculation
            # Define necessary constants and matrices for e-ph interaction
            nmode, nph_max = 1, 3
            omega = 0.1
            h1p = numpy.zeros((nmode, nmode))
            h1p[0, 0] = omega

            dd = numpy.ones(3)
            vv = dd * alph / numpy.linalg.norm(dd)
            vv = vv.reshape(nmode, 3)

            d_ao = m.intor('int1e_r', comp=3).reshape(3, nao, nao)
            h1e1p_ao = numpy.einsum('Ix,xpq->pqI', vv, d_ao)

            fci_obj = eph.FCI(m, mo=coeff_lo, h1p=h1p, h1e1p=h1e1p_ao, nph_max=nph_max, singlet=False)
            fci_obj.max_cycle = 1000
            fci_obj.conv_tol = 1e-6
            fci_obj.verbose = 10
            e0, c0 = fci_obj.kernel(nroots=nstate, ci0=ci0)

            singlet_count = 0
            triplet_count = 0

            for istate in range(nstate):
                ss = eph.fci.spin_square(c0[istate], norb, m.nelec, nmode, nph_max)[1]
                np = eph.fci.make_rdm1p(c0[istate], norb, m.nelec, nmode, nph_max)[0, 0]
                is_singlet = abs(ss - 1.0) < 1e-6
                is_triplet = abs(ss - 3.0) < 1e-6
                print("State %3d: energy = %12.6f, S^2 = %6.4f, N = %6.4f" % (istate, e0[istate], ss, np))
                print(singlet_count, triplet_count)
                assert is_singlet or is_triplet

                if is_singlet and singlet_count < nsinglet:
                    singlet_count += 1
                    singlet_data.append(e0[istate])
                    singlet_data.append(np)

                if is_triplet and triplet_count < ntriplet:
                    triplet_count += 1
                    triplet_data.append(e0[istate])
                    triplet_data.append(np)

                if singlet_count >= nsinglet and triplet_count >= ntriplet:
                    break

        tmp = "% 13.6f, " % bond_length
        for istate in range(nsinglet):
            tmp += "% 12.6f, % 12.6f, " % (singlet_data[2 * istate], singlet_data[2 * istate + 1])

        for istate in range(ntriplet):
            tmp += "% 12.6f, % 12.6f, " % (triplet_data[2 * istate], triplet_data[2 * istate + 1])

        tmp = tmp[:-2]
        print(tmp)
        out.write(tmp + "\n")

if __name__ == "__main__":
    # Define the bond lengths to be calculated
    basis_set = '631g'
    bond_lengths = numpy.linspace(2.2, 5.0, 29)
    main(mol="lih", basis_set=basis_set, bond_lengths=bond_lengths, alph=None)
    main(mol="lih", basis_set=basis_set, bond_lengths=bond_lengths, alph=0.001)
    main(mol="lih", basis_set=basis_set, bond_lengths=bond_lengths, alph=0.002)
    main(mol="lih", basis_set=basis_set, bond_lengths=bond_lengths, alph=0.003)
    main(mol="lih", basis_set=basis_set, bond_lengths=bond_lengths, alph=0.004)
    main(mol="lih", basis_set=basis_set, bond_lengths=bond_lengths, alph=0.005)
    
