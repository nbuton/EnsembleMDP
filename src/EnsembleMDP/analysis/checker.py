import numpy as np
import logging


def check_atom_consistency(u):
    """Checks if every frame in the trajectory has the same number of atoms."""
    n_topology = u.atoms.n_atoms
    inconsistent_frames = []

    logging.debug(f"Topology atom count: {n_topology}")

    for ts in u.trajectory:
        if u.atoms.n_atoms != n_topology:
            inconsistent_frames.append((ts.frame, u.atoms.n_atoms))

    if not inconsistent_frames:
        logging.debug("All frames are consistent.")
        return True
    else:
        for frame, count in inconsistent_frames:
            logging.debug(f"Frame {frame}: found {count} atoms (Expected {n_topology})")
        raise ValueError("Inconsistency detected!")


def verify_chain_continuity(traj, distance_threshold):
    """
    Checks for physical gaps between consecutive residues.
    Returns a list of indices where continuity is broken.
    """
    topology = traj.topology
    breaks = []

    # Identify CA atoms
    ca_indices = topology.select("name CA")

    # Calculate distances between consecutive CA atoms over the first frame
    for i in range(len(ca_indices) - 1):
        res_current = topology.atom(ca_indices[i]).residue
        res_next = topology.atom(ca_indices[i + 1]).residue

        # Only check if they are supposed to be in the same chain
        if res_current.chain == res_next.chain:
            # Measure distance (in nanometers)
            dist = np.linalg.norm(
                traj.xyz[0, ca_indices[i + 1], :] - traj.xyz[0, ca_indices[i], :]
            )

            if dist > distance_threshold:
                breaks.append(
                    {"res_i": res_current, "res_j": res_next, "distance": dist}
                )

    if not breaks:
        print("Chain is continuous.")
    else:
        for b in breaks:
            print(
                f"Break detected between {b['res_i']} and {b['res_j']}: {b['distance']:.3f} nm"
            )
        raise ValueError("Chain is not continious")

    return breaks


def verify_index_continuity(traj):
    """
    Checks for gaps in residue numbering (resSeq).
    Returns a list of gaps found.
    """
    topology = traj.topology
    index_gaps = []

    for chain in topology.chains:
        residues = list(chain.residues)
        for i in range(len(residues) - 1):
            res_current = residues[i]
            res_next = residues[i + 1]

            # Check if the next residue number is exactly current + 1
            expected_next = res_current.resSeq + 1
            actual_next = res_next.resSeq

            if actual_next != expected_next:
                gap_info = {
                    "chain": chain.index,
                    "before": res_current,
                    "after": res_next,
                    "expected": expected_next,
                    "actual": actual_next,
                }
                index_gaps.append(gap_info)

    if not index_gaps:
        logging.info("Residue numbering is continuous.")
    else:
        for gap in index_gaps:
            print(
                f"Index Gap in Chain {gap['chain']}: "
                f"Residue {gap['before']} (seq {gap['before'].resSeq}) is followed by "
                f"{gap['after']} (seq {gap['actual']}). Expected {gap['expected']}."
            )
        raise ValueError("Index continiuty is not correct")

    return index_gaps


def check_for_dssp_atoms(traj):
    """
    Checks if protein residues contain the H and O atoms required by DSSP.
    """
    topology = traj.topology
    missing_atoms = []

    for residue in topology.residues:
        # Skip non-protein residues (ligands, water, ions)
        if not residue.is_protein:
            continue

        atom_names = [atom.name for atom in residue.atoms]

        # DSSP needs Carbonyl Oxygen ('O') and Amide Hydrogen ('H')
        # Note: N-terminal residues and Proline are special cases,
        # but a total lack of 'H' atoms usually indicates a PDB-style structure.
        has_O = "O" in atom_names
        has_H = "H" in atom_names or "HN" in atom_names

        if not has_O or (not has_H and residue.name != "PRO"):
            missing_atoms.append((residue, "Missing H" if not has_H else "Missing O"))

    if not missing_atoms:
        logging.debug("All residues have required DSSP atoms.")
    else:
        logging.debug(f"Found {len(missing_atoms)} residues with missing atoms.")
        # Print the first 5 examples
        for res, issue in missing_atoms[:5]:
            logging.debug(f"  - {res}: {issue}")
        raise ValueError("Missing atoms")

    return missing_atoms


def verify_protein_purity(md_traj):
    """
    Identifies non-protein residues that could cause shape mismatches
    in dihedral or DSSP calculations.
    """
    topology = md_traj.topology
    non_protein_residues = []

    for residue in topology.residues:
        # MDTraj uses 'is_protein' to check for standard amino acids
        if not residue.is_protein:
            non_protein_residues.append(
                {
                    "index": residue.index,
                    "name": residue.name,
                    "chain": residue.chain.index,
                    "n_atoms": residue.n_atoms,
                }
            )

    if not non_protein_residues:
        logging.debug("Topology contains only protein residues.")
        return True
    else:
        print(f"Found {len(non_protein_residues)} non-protein residues.")
        for res in non_protein_residues:
            print(
                f"  - [{res['name']}] at index {res['index']} (Chain {res['chain']}) "
                f"with {res['n_atoms']} atoms"
            )

        # This explains your broadcasting error:
        # n_residues (used for array shape) counts these,
        # but compute_phi ignores them.
        raise ValueError("Protein is not pure")
