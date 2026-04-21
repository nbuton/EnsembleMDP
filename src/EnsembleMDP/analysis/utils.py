import numpy as np
import pandas as pd
import ast
import tempfile
from pathlib import Path
import tarfile
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import MDAnalysis as mda
import logging


def get_ensemble_summary(
    data_dict, include_min_max=False, include_histogram=False, bins=20
):
    """
    Summarizes time-series data into ensemble-wide descriptors.

    By default, this method computes the arithmetic mean and standard deviation
    for every property provided in the data_dict,

    Args:
        data_dict (dict): Dictionary of property arrays.
        include_min_max (bool): If True, returns min and max values.
        include_histogram (bool): If True, returns counts and bin edges.
        bins (int/str): Number of bins or method (e.g., 'auto') for np.histogram.
    """
    summary = {}

    for key, values in data_dict.items():
        data = np.array(values)

        # Core Stats
        summary[f"{key}_mean"] = np.mean(data)
        summary[f"{key}_std"] = np.std(data)

        # Optional: Range
        if include_min_max:
            summary[f"{key}_min"] = np.min(data)
            summary[f"{key}_max"] = np.max(data)

        # Optional: Distribution
        if include_histogram:
            # density=True gives the probability density instead of raw counts
            counts, bin_edges = np.histogram(data, bins=bins, density=True)
            summary[f"{key}_hist_counts"] = counts
            summary[f"{key}_hist_edges"] = bin_edges

    return summary


def mean_std(array, squared: bool):
    if squared:
        new_array = array**2
    else:
        new_array = array
    return np.mean(new_array), np.std(new_array)


def get_sequence_from_mdtraj(md_traj):
    """
    Extracts the amino acid sequence as a string of single-letter codes.
    """
    # Dictionary to map 3-letter codes to 1-letter codes
    d3to1 = {
        "CYS": "C",
        "ASP": "D",
        "SER": "S",
        "GLN": "Q",
        "LYS": "K",
        "ILE": "I",
        "PRO": "P",
        "THR": "T",
        "PHE": "F",
        "ASN": "N",
        "GLY": "G",
        "HIS": "H",
        "LEU": "L",
        "ARG": "R",
        "TRP": "W",
        "ALA": "A",
        "VAL": "V",
        "GLU": "E",
        "TYR": "Y",
        "MET": "M",
    }

    # Extract residues from the topology
    residues = [res.name for res in md_traj.topology.residues]

    # Convert to 1-letter codes, defaulting to 'X' for unknown/non-standard
    sequence = "".join([d3to1.get(res, "X") for res in residues])

    return sequence


import gzip
import tarfile
import shutil
import tempfile
from pathlib import Path


def decompress_pdb_path(compressed_pdb_path):
    compressed_pdb_path = Path(compressed_pdb_path)

    # 1. Check for GZIP magic number (0x1f 0x8b)
    with open(compressed_pdb_path, "rb") as f:
        header = f.read(2)

    if header == b"\x1f\x8b":
        temp_dir = tempfile.TemporaryDirectory()

        # 2. Check if it's a TAR archive (even if compressed)
        if tarfile.is_tarfile(compressed_pdb_path):
            print(f"Detected TAR archive: {compressed_pdb_path.name}")
            try:
                with tarfile.open(compressed_pdb_path, "r:gz") as tar:
                    # Find the first member ending in .pdb
                    members = [
                        m for m in tar.getmembers() if m.name.lower().endswith(".pdb")
                    ]
                    if not members:
                        temp_dir.cleanup()
                        raise FileNotFoundError(
                            "No .pdb file found inside the TAR archive."
                        )

                    tar.extract(members[0], path=temp_dir.name)
                    uncompressed_filepath = Path(temp_dir.name) / members[0].name
                    return uncompressed_filepath, temp_dir
            except Exception as e:
                temp_dir.cleanup()
                raise RuntimeError(f"Failed to extract TAR.GZ: {e}")

        # 3. If it's GZIP but NOT a TAR, treat it as a simple compressed file
        else:
            print(f"Detected simple GZIP file: {compressed_pdb_path.name}")
            # Use .stem to remove one layer of extension, or default to a generic name
            # since you mentioned extensions are unreliable.
            output_name = compressed_pdb_path.name.replace(".gz", "")
            if not output_name.lower().endswith(".pdb"):
                output_name += ".pdb"

            uncompressed_filepath = Path(temp_dir.name) / output_name

            try:
                with gzip.open(compressed_pdb_path, "rb") as f_in:
                    with open(uncompressed_filepath, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return uncompressed_filepath, temp_dir
            except Exception as e:
                temp_dir.cleanup()
                raise RuntimeError(f"Failed to decompress GZIP: {e}")

    # Not a GZIP file, return as is
    return compressed_pdb_path, None


import MDAnalysis as mda
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from openmm import Vec3
import openmm.unit as unit
import numpy as np
import io
import tempfile
from pathlib import Path


# Helper to prevent MDAnalysis PDBWriter from closing our StringIO
class NonClosingStringIO(io.StringIO):
    """StringIO that ignores .close() calls from MDAnalysis writers."""

    def close(self):
        pass  # MDAnalysis calls close() on exit — we keep the buffer readable for PDBFixer


def get_corrected_pdb(compressed_input_path):
    current_pdb_path, decomp_temp = decompress_pdb_path(compressed_input_path)
    try:
        # 1. Load the original universe to iterate through frames
        u_orig = mda.Universe(str(current_pdb_path))
        n_frames = len(u_orig.trajectory)

        save_temp = tempfile.TemporaryDirectory()
        corrected_path = Path(save_temp.name) / f"corrected_{current_pdb_path.name}"

        logging.info(f"Fixing {n_frames} frames for {current_pdb_path.name}...")

        # === TOPOLOGY FIX ONCE (frame 0) + ROBUST ATOM MAPPING + NUMPY QUANTITY ===
        u_orig.trajectory[0]  # ensure we are on frame 0
        frame_buffer = NonClosingStringIO()
        with mda.Writer(frame_buffer, format="PDB") as W:
            W.write(u_orig.atoms)
        frame_buffer.seek(0)

        f = PDBFixer(pdbfile=frame_buffer)
        f.findMissingResidues()
        f.findMissingAtoms()
        f.addMissingAtoms()
        f.addMissingHydrogens(7.0)  # required to eliminate "Missing atoms"

        fixed_topology = f.topology
        fixed_positions_ref = f.positions  # OpenMM Quantity or list[Vec3] (in nm)

        # Build robust mapping using PDB metadata (chain + resid + atom name)
        orig_atom_by_key = {}
        for atom in u_orig.atoms:
            chain = getattr(atom, "chainID", getattr(atom, "segid", "")) or ""
            res_id = str(atom.resid)
            key = (chain, res_id, atom.name)
            orig_atom_by_key[key] = atom.index

        fixed_to_orig_idx = []
        for atom in fixed_topology.atoms():
            chain = atom.residue.chain.id if atom.residue.chain else ""
            res_id = atom.residue.id
            key = (chain, res_id, atom.name)
            orig_idx = orig_atom_by_key.get(key)
            fixed_to_orig_idx.append(orig_idx)  # None = added atom (H or missing heavy)

        # n_fixed = number of atoms in the FINAL topology (original + added)
        n_fixed = len(fixed_to_orig_idx)

        # Robust conversion of reference positions (frame 0) to numpy array in nm
        # Works for both modern OpenMM (Quantity) and older OpenMM (list of Vec3)
        if hasattr(fixed_positions_ref, "value_in_unit"):
            fixed_ref_nm = fixed_positions_ref.value_in_unit(unit.nanometer)
        else:
            fixed_ref_nm = np.array(
                [[p.x, p.y, p.z] for p in fixed_positions_ref], dtype=np.float64
            )
        fixed_ref_nm = np.asarray(fixed_ref_nm, dtype=np.float64).reshape(-1, 3)

        # 3. Write fixed multi-model PDB
        with open(corrected_path, "w") as f_out:
            for i, ts in enumerate(u_orig.trajectory):
                # Current original positions (Å)
                orig_pos_ang = u_orig.atoms.positions

                # Build frame positions as numpy array (nm)
                frame_pos_nm = np.empty((n_fixed, 3), dtype=np.float64)
                for m, orig_idx in enumerate(fixed_to_orig_idx):
                    if orig_idx is not None:
                        p = orig_pos_ang[orig_idx]
                        frame_pos_nm[m] = (p[0] / 10.0, p[1] / 10.0, p[2] / 10.0)
                    else:
                        frame_pos_nm[m] = fixed_ref_nm[m]

                # Convert to proper OpenMM Quantity (exactly what PDBFile expects)
                frame_positions = unit.Quantity(frame_pos_nm, unit.nanometer)

                # Write this frame
                f_out.write(f"MODEL {i+1}\n")
                PDBFile.writeFile(fixed_topology, frame_positions, f_out, keepIds=True)
                f_out.write("ENDMDL\n")

        if decomp_temp:
            decomp_temp.cleanup()
        return corrected_path, save_temp

    except Exception as e:
        if decomp_temp:
            decomp_temp.cleanup()
        raise RuntimeError(f"Multi-frame PDBFixer failed: {e}") from e
