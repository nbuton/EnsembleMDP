import os
import subprocess
import mdtraj as md


class CGBackmapper:
    """
    Backmaps Coarse-Grained (CG) trajectories to all-atom (AA) representation.
    """

    DEFAULT_RESIDUE_MAPPING = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }

    def __init__(
        self,
        top_file,
        traj_file,
        cg_model="CalphaBasedModel",
        device="cpu",
        cg2all_batch_size=16,
        cg2all_nb_proc=4,
        custom_mapping=None,
        force_ca_rename=False,
    ):
        self.top_file = os.path.abspath(top_file)
        self.traj_file = os.path.abspath(traj_file)
        self.cg_model = cg_model
        self.device = device.lower()
        self.cg2all_batch_size = cg2all_batch_size
        self.cg2all_nb_proc = cg2all_nb_proc
        self.force_ca_rename = force_ca_rename

        self.residue_mapping = self.DEFAULT_RESIDUE_MAPPING.copy()
        if custom_mapping:
            self.residue_mapping.update(custom_mapping)

    def run(self, preprocess=False, out_format="dcd"):
        base_dir = os.path.dirname(self.top_file)
        aa_pdb = os.path.join(base_dir, "aa_topology.pdb")
        aa_traj = os.path.join(base_dir, f"aa_traj.{out_format.strip('.')}")

        if os.path.exists(aa_pdb) and os.path.exists(aa_traj):
            print(f"Skipping: All-atom files already exist in {base_dir}")
            return

        cg2all_in_pdb, cg2all_in_traj = self.top_file, self.traj_file
        temp_files = []

        try:
            if preprocess:
                fixed_pdb = os.path.join(base_dir, "temp_fixed_cg.pdb")
                fixed_traj = os.path.join(base_dir, "temp_fixed_cg.dcd")
                temp_files.extend([fixed_pdb, fixed_traj])
                self._fix_topology(fixed_pdb, fixed_traj)
                cg2all_in_pdb, cg2all_in_traj = fixed_pdb, fixed_traj

            temp_aa_dcd = os.path.join(base_dir, "temp_aa.dcd")
            temp_files.append(temp_aa_dcd)

            self._reconstruct(cg2all_in_pdb, cg2all_in_traj, aa_pdb, temp_aa_dcd)
            self._convert_format(temp_aa_dcd, aa_pdb, aa_traj)

        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)

    def _fix_topology(self, out_pdb, out_traj):
        t = md.load(self.traj_file, top=self.top_file)
        fixed_top = md.Topology()
        chain_map = {chain.index: fixed_top.add_chain() for chain in t.top.chains}

        for atom in t.top.atoms:
            old_res = atom.residue.name
            new_res = self.residue_mapping.get(
                old_res.upper() if len(old_res) == 1 else old_res, old_res
            )
            res = fixed_top.add_residue(new_res, chain_map[atom.residue.chain.index])

            if self.force_ca_rename:
                fixed_top.add_atom("CA", element=md.element.carbon, residue=res)
            else:
                fixed_top.add_atom(atom.name, element=atom.element, residue=res)

        fixed_traj = md.Trajectory(
            t.xyz, fixed_top, t.time, t.unitcell_lengths, t.unitcell_angles
        )
        fixed_traj[0].save_pdb(out_pdb)
        fixed_traj.save_dcd(out_traj)

    def _reconstruct(self, in_pdb, in_traj, out_pdb, out_traj):
        cmd = [
            "convert_cg2all",
            "-p",
            in_pdb,
            "-d",
            in_traj,
            "-o",
            out_traj,
            "-opdb",
            out_pdb,
            "--cg",
            self.cg_model,
            "--batch",
            str(self.cg2all_batch_size),
            "--proc",
            str(self.cg2all_nb_proc),
            "--device",
            self.device,
        ]
        subprocess.run(cmd, check=True)

    def _convert_format(self, in_traj, top_pdb, out_traj):
        if out_traj.endswith(".dcd"):
            os.rename(in_traj, out_traj)
        else:
            traj = md.load(in_traj, top=top_pdb)
            traj.save(out_traj)
