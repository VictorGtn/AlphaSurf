import multiprocessing
import os
import platform
import warnings

import hydra
import numpy as np
import torch
from alphasurf.protein.create_esm import get_esm_embedding_batch
from alphasurf.protein.surfaces import SurfaceObject
from alphasurf.utils.data_utils import PreprocessDataset
from alphasurf.utils.python_utils import do_all
from alphasurf.utils.timing_stats import TimingStats
from omegaconf import DictConfig
from torch.utils.data import Dataset

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)


class PreprocessPatchDataset(Dataset):
    """
    Dataset for preprocessing MaSIF-Ligand patches into surface objects.
    """

    def __init__(
        self, data_dir=None, recompute=False, face_reduction_rate=1.0, use_pymesh=False
    ):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if data_dir is None:
            masif_ligand_data_dir = os.path.join(
                script_dir, "..", "..", "..", "data", "masif_ligand"
            )
        else:
            masif_ligand_data_dir = data_dir
        self.patch_dir = os.path.join(masif_ligand_data_dir, "dataset_MasifLigand")
        self.out_surf_dir_ours = os.path.join(
            masif_ligand_data_dir, f"surf_{face_reduction_rate}_{use_pymesh}"
        )

        self.face_reduction_rate = face_reduction_rate
        self.use_pymesh = use_pymesh

        self.patches = list(os.listdir(self.patch_dir))
        self.recompute = recompute
        os.makedirs(self.out_surf_dir_ours, exist_ok=True)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        path_torch_name = patch.replace(".npz", ".pt")
        surface_ours_dump = os.path.join(self.out_surf_dir_ours, path_torch_name)
        try:
            patch_in = os.path.join(self.patch_dir, patch)
            data = np.load(patch_in, allow_pickle=True)
            verts = data["pkt_verts"]
            faces = data["pkt_faces"].astype(int)

            if self.recompute or not os.path.exists(surface_ours_dump):
                surface_ours = SurfaceObject.from_verts_faces(
                    verts=verts,
                    faces=faces,
                    face_reduction_rate=self.face_reduction_rate,
                    use_pymesh=self.use_pymesh,
                )
                surface_ours.add_geom_feats()
                surface_ours.save_torch(surface_ours_dump)
            success = 1
        except Exception as e:
            print(e)
            success = 0
        return success


class PreProcessPDBDataset(PreprocessDataset):
    """
    Dataset for preprocessing MaSIF-Ligand PDB files into surface objects.

    Supports generating multiple augmented views per protein using noise-based
    data augmentation. Augmentation is handled by the parent PreprocessDataset class.
    """

    def __init__(
        self,
        data_dir=None,
        compute_s=True,
        recompute_s=False,
        recompute_g=False,
        max_vert_number=1000000,
        face_reduction_rate=1.0,
        use_pymesh=True,
        surface_method="msms",
        sbl_exe_path=None,
        alpha_value=0.1,
        n_augmented_views=1,
        augmentation_sigma=0.3,
        augmentation_noise_type="normal",
    ):
        if data_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(
                script_dir, "..", "..", "..", "data", "masif_ligand"
            )

        # Pass augmentation params to parent class (skip dir creation so we can override it)
        super().__init__(
            data_dir=data_dir,
            recompute_s=recompute_s,
            recompute_g=recompute_g,
            max_vert_number=max_vert_number,
            face_reduction_rate=face_reduction_rate,
            surface_method=surface_method,
            sbl_exe_path=sbl_exe_path,
            alpha_value=alpha_value,
            n_augmented_views=n_augmented_views,
            augmentation_sigma=augmentation_sigma,
            augmentation_noise_type=augmentation_noise_type,
            _skip_surf_dir_creation=True,
        )

        # Override PDB location for MaSIF-Ligand specific structure
        self.pdb_dir = os.path.join(data_dir, "raw_data_MasifLigand", "pdb")

        self.compute_s = compute_s
        if self.compute_s:
            # Build descriptive surface directory name
            if surface_method == "alpha_complex":
                surface_dirname = (
                    f"surfaces_full_alpha{alpha_value}_fr{face_reduction_rate}"
                )
            else:
                surface_dirname = f"surfaces_full_msms_fr{face_reduction_rate}"
            if use_pymesh:
                surface_dirname += "_pymesh"
            if n_augmented_views > 1:
                surface_dirname += f"_aug{n_augmented_views}_{augmentation_noise_type}_sigma{augmentation_sigma}"
            self.out_surf_dir = os.path.join(data_dir, surface_dirname)
            os.makedirs(self.out_surf_dir, exist_ok=True)
        else:
            # remove dir if not computing surfaces
            surface_dirname = f"surfaces_{face_reduction_rate}{f'_{use_pymesh}' if use_pymesh is not None else ''}"
            out_surf_dir = os.path.join(data_dir, surface_dirname)
            try:
                os.rmdir(out_surf_dir)
            except (FileNotFoundError, OSError):
                pass

        self.all_pdbs = self.get_all_pdbs()

    def __getitem__(self, idx):
        pdb = self.all_pdbs[idx]
        name = pdb[0:-4]
        if self.compute_s:
            success = self.name_to_surf_graphs(name)
        else:
            success = self.name_to_graphs(name)
        return success


def get_surface_tool_path(cfg, script_dir):
    """
    Auto-detect platform and return appropriate surface tool executable path.

    Returns the path from config if set, otherwise auto-detects based on platform.
    """
    surface_method = cfg.preprocessing.surface_method

    if surface_method == "msms":
        # Check if path is explicitly set in config
        if cfg.preprocessing.get("msms_exe_path") and cfg.preprocessing.msms_exe_path:
            return cfg.preprocessing.msms_exe_path

        # Auto-detect platform
        system = platform.system()
        if system == "Darwin":  # macOS
            platform_dir = "msms_macos"
        elif system == "Linux":
            platform_dir = "msms_linux"
        elif system == "Windows":
            platform_dir = "msms_windows"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        msms_path = os.path.normpath(
            os.path.join(script_dir, "..", "..", "..", "bin", platform_dir, "msms")
        )

        if not os.path.exists(msms_path):
            raise FileNotFoundError(
                f"MSMS binary not found at {msms_path}. "
                f"Please install MSMS binaries for {system} in bin/{platform_dir}/"
            )

        return msms_path

    elif surface_method == "alpha_complex":
        # Check if path is explicitly set in config
        if cfg.preprocessing.sbl_exe_path:
            return cfg.preprocessing.sbl_exe_path

        # Alpha_complex binary is cross-platform, no need for platform-specific paths
        return None

    return None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    recompute = False
    recompute_s = True
    recompute_g = False
    use_pymesh = False

    # Resolve data_dir relative to the script's location to make it CWD-independent
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, cfg.data_dir))

    # Auto-detect surface tool path based on platform
    surface_tool_path = get_surface_tool_path(cfg, script_dir)

    # Auto-detect appropriate worker count (respect system limits)
    num_workers = min(cfg.loader.num_workers, multiprocessing.cpu_count())

    TimingStats.reset()

    # Preprocess patches (no augmentation - these are pocket patches)
    """dataset = PreprocessPatchDataset(recompute=recompute,
                                     face_reduction_rate=cfg.preprocessing.face_reduction_rate,
                                     use_pymesh=use_pymesh,
                                     data_dir=data_dir)
    do_all(dataset, num_workers=num_workers)"""

    # Preprocess full surfaces from PDB (with augmentation support)
    dataset = PreProcessPDBDataset(
        recompute_g=recompute_g,
        recompute_s=recompute_s,
        compute_s=True,
        face_reduction_rate=cfg.preprocessing.face_reduction_rate,
        use_pymesh=use_pymesh,
        surface_method=cfg.preprocessing.surface_method,
        sbl_exe_path=surface_tool_path,
        alpha_value=cfg.preprocessing.alpha_value,
        data_dir=data_dir,
        n_augmented_views=cfg.cfg_surface.n_augmented_views,
        augmentation_sigma=cfg.cfg_surface.augmentation_sigma,
        augmentation_noise_type=cfg.cfg_surface.augmentation_noise_type,
    )
    do_all(dataset, num_workers=num_workers)

    TimingStats.get().print_summary()

    masif_ligand_data_dir = data_dir
    pdb_dir = os.path.join(masif_ligand_data_dir, "raw_data_MasifLigand", "pdb")
    out_esm_dir = os.path.join(masif_ligand_data_dir, "esm")
    get_esm_embedding_batch(in_pdbs_dir=pdb_dir, dump_dir=out_esm_dir, batch_size=4)


if __name__ == "__main__":
    main()
