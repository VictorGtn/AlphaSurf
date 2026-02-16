import os
import sys

import numpy as np
import scipy.sparse

# Add project root to path
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

from alphasurf.protein.create_operators import compute_operators


def create_simple_mesh():
    # Create a simple 4x4 grid mesh (16 vertices) to satisfy the size requirements
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 4)
    xx, yy = np.meshgrid(x, y)
    verts = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()))).astype(
        np.float32
    )

    faces = []
    for i in range(3):
        for j in range(3):
            # Two triangles for each grid square
            # Indices:
            # v0 -- v1
            # |   /  |
            # v2 -- v3
            idx = i * 4 + j
            faces.append([idx, idx + 1, idx + 4])
            faces.append([idx + 1, idx + 5, idx + 4])

    faces = np.array(faces, dtype=np.int32)
    return verts, faces


def test_laplacian_integration():
    print("Testing Tufted Laplacian Integration...")

    verts, faces = create_simple_mesh()
    print(f"Created simple mesh with {len(verts)} verts and {len(faces)} faces")

    # 1. Compute Standard Laplacian via our modified function
    print("\n--- Standard Laplacian ---")
    try:
        _, mass_std, L_std, evals_std, _, _, _ = compute_operators(
            verts, faces, k_eig=1, use_robust_laplacian=False
        )
        print("Successfully computed standard operators")
        print(f"Mass shape: {mass_std.shape}, L shape: {L_std.shape}")
        if scipy.sparse.issparse(mass_std):
            print(
                "Mass is sparse matrix (as expected from new logic for consistency?) No, typically array or diag."
            )
            # check type
            print(f"Mass type: {type(mass_std)}")
    except Exception as e:
        print(f"FAILED Standard: {e}")
        import traceback

        traceback.print_exc()

    # 2. Compute Robust Laplacian via our modified function
    print("\n--- Robust (Tufted) Laplacian ---")
    try:
        _, mass_rob, L_rob, evals_rob, _, _, _ = compute_operators(
            verts, faces, k_eig=1, use_robust_laplacian=True
        )
        print("Successfully computed robust operators")
        print(f"Mass shape: {mass_rob.shape}, L shape: {L_rob.shape}")
        print(f"Mass type: {type(mass_rob)}")

        # Check if they are similar (on this simple manifold mesh, they should be identical)
        diff_L = np.abs((L_std - L_rob).data).max() if (L_std - L_rob).nnz > 0 else 0

        # Robust laplacian returns a diagonal matrix for mass, while standard logic used to return a vector
        # My modification made them both return sparse matrices (or handled it in eigsh call)
        # Let's check what came back.

        print(f"Max difference in L (Standard vs Robust): {diff_L}")

        if diff_L < 1e-5:
            print("SUCCESS: Standard and Robust Laplacians match on manifold mesh")
        else:
            print(
                "WARNING: Standard and Robust Laplacians differ significantly (might be expected for degenerate, but not here?)"
            )

    except ImportError:
        print("Skipped Robust test: robust_laplacian not installed")
    except Exception as e:
        print(f"FAILED Robust: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_laplacian_integration()
