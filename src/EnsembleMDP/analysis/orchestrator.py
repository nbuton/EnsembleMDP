import MDAnalysis
import mdtraj as md
import logging
from idpmdp.analysis.checker import (
    check_atom_consistency,
    check_for_dssp_atoms,
    verify_protein_purity,
    verify_index_continuity,
)
from idpmdp.analysis.utils import (
    get_corrected_pdb,
    get_ensemble_summary,
    mean_std,
)
from idpmdp.analysis.global_metrics import (
    compute_end_to_end_distance,
    compute_gyration_tensor_properties,
    compute_maximum_diameter,
    compute_scaling_exponent,
)
from idpmdp.analysis.residue_level_metrics import (
    compute_dihedral_sequence_tracks,
    compute_residue_sasa,
    compute_secondary_structure_propensities,
)
from idpmdp.analysis.matrix_metrics import (
    compute_contact_map,
    compute_dccm,
    compute_distance_fluctuations,
)


class ProteinAnalyzer:
    def __init__(
        self,
        pdb_path,
        xtc_path=None,
        frame_offset=0,  # For some data like IDRome the first 10 frames should be discarded
        n_subsample_trajectory=-1,
    ):
        """Initializes the Universe and checks the system size."""
        assert pdb_path.exists(), f"Expected file {pdb_path} to exist, but it does not."

        self.pdb_path, _ = get_corrected_pdb(
            pdb_path
        )  # Automatically decompress if it is a pdb.gz and also correct if missing atoms

        self.xtc_path = xtc_path
        topology = pdb_path
        trajectory = xtc_path  # This can be None, a string, or a list

        if trajectory is None:
            if n_subsample_trajectory != -1:
                raise NotImplementedError()
            self.md_analysis_u = MDAnalysis.Universe(topology)
            check_atom_consistency(self.md_analysis_u)
            self.md_traj = md.load(self.pdb_path)
            logging.info(
                f"Loaded Universe with {len(self.md_analysis_u.trajectory)} frames."
            )
            assert (
                len(self.md_analysis_u.trajectory) > 1
            ), "Be carefull no trajectory is loaded. Delete this raise if you known what you do"

        else:
            assert (
                xtc_path.exists()
            ), f"Expected file {xtc_path} to exist, but it does not."
            if n_subsample_trajectory != -1:
                self.md_analysis_u = MDAnalysis.Universe(
                    topology,
                    trajectory,
                )

            else:
                self.md_analysis_u = MDAnalysis.Universe(
                    topology,
                    trajectory,
                )

            check_atom_consistency(self.md_analysis_u)
            self.md_traj = md.load(self.xtc_path, top=self.pdb_path)

        if n_subsample_trajectory != -1:
            self.md_analysis_u.transfer_to_memory(
                start=frame_offset, stop=frame_offset + n_subsample_trajectory
            )
            self.md_traj = self.md_traj[
                frame_offset : n_subsample_trajectory + frame_offset
            ]
        else:
            self.md_analysis_u.transfer_to_memory(start=frame_offset)
            self.md_traj = self.md_traj[frame_offset:]

        logging.info(f"New frames: {self.md_analysis_u.trajectory.n_frames}")
        verify_index_continuity(self.md_traj)
        check_for_dssp_atoms(self.md_traj)
        verify_protein_purity(self.md_traj)

        n_chains = len(set(self.md_analysis_u.atoms.chainIDs))
        logging.info(f"Number of chains: {n_chains}")
        assert n_chains == 1, "There are multiple chains in this pdb"

        # Select only protein atoms
        protein_selection = self.md_analysis_u.select_atoms("protein")
        self.md_analysis_u.transfer_to_memory(atomgroup=protein_selection)
        logging.info(f"Total frames captured: {self.md_analysis_u.trajectory.n_frames}")

        assert hasattr(self.md_analysis_u.atoms, "elements"), "No elemnts attributes"
        assert hasattr(self.md_analysis_u.atoms, "masses"), "No masses attributes"
        # Verify that the universe contains only one segment
        assert (
            len(self.md_analysis_u.segments) == 1
        ), "The provided PDB file contains multiple segments."

        # Identify the protein and its size
        heavy_atoms = self.md_analysis_u.select_atoms("not element H")
        self.is_coarse_grained = len(heavy_atoms) < (
            len(self.md_analysis_u.residues) * 4
        )
        self.protein_atoms = self.md_analysis_u.select_atoms("protein")
        self.residues = self.protein_atoms.residues
        self.protein_size = len(self.residues)

        logging.info(
            f"Loaded Universe with {len(self.md_analysis_u.trajectory)} frames."
        )
        logging.info(f"Protein size: {self.protein_size} residues.")

    def compute_all(
        self,
        sasa_n_sphere=960,
        scaling_min_sep=5,
        contact_cutoff=8.0,
    ):
        """
        Executes all analysis methods and aggregates results into a single dictionary.

        Returns:
            dict: Comprehensive analysis results.
        """
        results = {}
        logging.debug("Start computing end to end distances")
        results["avg_squared_Ree"], results["std_squared_Ree"] = mean_std(
            compute_end_to_end_distance(self.md_analysis_u, self.residues), squared=True
        )
        logging.debug("Start computing maximum diameter")
        results["avg_maximum_diameter"], results["std_maximum_diameter"] = mean_std(
            compute_maximum_diameter(self.md_analysis_u), squared=False
        )
        logging.debug("Start computing gyration tensor")
        gyration_output = compute_gyration_tensor_properties(self.md_traj)
        results.update(
            get_ensemble_summary(
                gyration_output, include_min_max=False, include_histogram=False, bins=20
            )
        )

        logging.debug("Start computing scalling exponent")
        results["scaling_exponent"] = compute_scaling_exponent(
            self.md_traj, min_s=scaling_min_sep
        )

        results.update(compute_secondary_structure_propensities(self.md_traj))

        logging.debug("Start computing DCCM")
        results["dccm"] = compute_dccm(self.md_analysis_u, self.protein_atoms)
        results.update(compute_dihedral_sequence_tracks(self.md_traj, bins=60))

        logging.debug("Start computing distance fluctuation")
        results["distance_fluctuations"] = compute_distance_fluctuations(
            self.md_analysis_u, self.protein_atoms
        )
        logging.debug("Start computing contact map frequency")
        results["contact_map"] = compute_contact_map(
            self.md_analysis_u, self.protein_atoms, cutoff=contact_cutoff
        )

        logging.debug("Start computing SASA")
        results.update(
            compute_residue_sasa(self.md_traj, n_sphere_points=sasa_n_sphere)
        )

        return results
