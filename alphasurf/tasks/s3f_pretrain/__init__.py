"""
S3F-style pretraining for AlphaSurf.

Masked residue-type prediction on CATH v4.3.0, mirroring S3F (Zhang et al.,
NeurIPS 2024) but with the AlphaSurf encoder (ProNet graph + DiffusionNet
surface + GVP fusion) replacing S3F's GVP-GNN, and AlphaSurf's CGAL
alpha-complex mesh replacing S3F's dMaSIF point cloud. ESM2-650M is frozen
and run on the masked sequence per batch to provide initial node features.
"""
