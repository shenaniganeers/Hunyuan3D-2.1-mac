"""Unit tests for the Metal custom rasterizer kernel."""
import sys
import torch
sys.path.insert(0, ".")
import custom_rasterizer_kernel
import pytest


def make_triangle_verts(v0, v1, v2):
    """Create a vertex tensor (N, 4) with homogeneous coordinates.

    Each vertex is (x, y, z, w) in clip space:
      x/w in [-1, 1]  -> maps to [0, width-1]
      y/w in [-1, 1]  -> maps to [0, height-1]
      z/w              -> depth (higher = farther, mapped to ~[0,1])
    """
    verts = torch.tensor([v0, v1, v2], dtype=torch.float32)
    return verts


def make_faces(*faces):
    """Create a face tensor (M, 3) of int32 vertex indices."""
    return torch.tensor(faces, dtype=torch.int32)


class TestSingleTriangle:
    """Rasterize a single triangle covering the centre of a small image."""

    def setup_method(self):
        # A triangle in clip space whose projection covers roughly the middle
        # of a 64x64 image.  w=1 so clip coords == NDC.
        #   v0 = centre-top, v1 = bottom-left, v2 = bottom-right
        self.V = make_triangle_verts(
            [0.0, 0.5, 0.5, 1.0],   # top centre
            [-0.5, -0.5, 0.5, 1.0], # bottom-left
            [0.5, -0.5, 0.5, 1.0],  # bottom-right
        )
        self.F = make_faces([0, 1, 2])
        self.D = torch.zeros(0, dtype=torch.float32)
        self.W, self.H = 64, 64

    def test_produces_nonzero_face_indices(self):
        findices, bary = custom_rasterizer_kernel.rasterize_image(
            self.V, self.F, self.D, self.W, self.H, 1e-6, 0)
        assert findices.shape == (self.H, self.W)
        assert bary.shape == (self.H, self.W, 3)
        # At least some pixels should be covered (face index > 0)
        assert (findices > 0).any(), "No pixels were rasterized"

    def test_barycentric_sum_approximately_one(self):
        findices, bary = custom_rasterizer_kernel.rasterize_image(
            self.V, self.F, self.D, self.W, self.H, 1e-6, 0)
        covered = findices > 0
        if covered.any():
            bary_sum = bary[covered].sum(dim=-1)
            assert torch.allclose(bary_sum, torch.ones_like(bary_sum), atol=1e-3), \
                f"Barycentric sums deviate from 1.0: min={bary_sum.min()}, max={bary_sum.max()}"

    def test_barycentric_nonnegative(self):
        findices, bary = custom_rasterizer_kernel.rasterize_image(
            self.V, self.F, self.D, self.W, self.H, 1e-6, 0)
        covered = findices > 0
        if covered.any():
            assert (bary[covered] >= -1e-6).all(), \
                f"Negative barycentric values found: min={bary[covered].min()}"


class TestDepthOrdering:
    """A nearer triangle should occlude a farther one at overlapping pixels."""

    def test_nearer_occludes_farther(self):
        # Two triangles at different depths, both covering image centre.
        # Triangle 0: depth 0.8 (farther)
        # Triangle 1: depth 0.3 (nearer)
        V = make_triangle_verts(
            [0.0, 0.5, 0.8, 1.0],
            [-0.5, -0.5, 0.8, 1.0],
            [0.5, -0.5, 0.8, 1.0],
        )
        V_near = make_triangle_verts(
            [0.0, 0.5, 0.3, 1.0],
            [-0.5, -0.5, 0.3, 1.0],
            [0.5, -0.5, 0.3, 1.0],
        )
        V_all = torch.cat([V, V_near], dim=0)
        F = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
        D = torch.zeros(0, dtype=torch.float32)

        findices, bary = custom_rasterizer_kernel.rasterize_image(
            V_all, F, D, 64, 64, 1e-6, 0)

        covered = findices > 0
        assert covered.any(), "No pixels covered by either triangle"
        # Face indices use 1-based indexing. The nearer triangle is face index 2.
        # All covered pixels should show the nearer face.
        covered_indices = findices[covered]
        assert (covered_indices == 2).all(), \
            f"Expected all covered pixels to show face 2 (nearer), got unique values: {covered_indices.unique()}"


class TestEmptyInput:
    """Empty face list should not crash."""

    def test_zero_faces(self):
        V = torch.zeros((3, 4), dtype=torch.float32)
        F = torch.zeros((0, 3), dtype=torch.int32)
        D = torch.zeros(0, dtype=torch.float32)
        findices, bary = custom_rasterizer_kernel.rasterize_image(
            V, F, D, 32, 32, 1e-6, 0)
        assert findices.shape == (32, 32)
        assert (findices == 0).all()


class TestLargerResolution:
    """512x512 should work without crashes."""

    def test_512x512(self):
        V = make_triangle_verts(
            [0.0, 0.5, 0.5, 1.0],
            [-0.5, -0.5, 0.5, 1.0],
            [0.5, -0.5, 0.5, 1.0],
        )
        F = make_faces([0, 1, 2])
        D = torch.zeros(0, dtype=torch.float32)
        findices, bary = custom_rasterizer_kernel.rasterize_image(
            V, F, D, 512, 512, 1e-6, 0)
        assert findices.shape == (512, 512)
        assert bary.shape == (512, 512, 3)
        assert (findices > 0).any(), "No pixels rasterized at 512x512"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
