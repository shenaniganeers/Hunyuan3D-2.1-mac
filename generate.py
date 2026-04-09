#!/usr/bin/env python3
"""Generate a textured 3D model from a single image.

Usage:
    python generate.py photo.png -o model.glb
    python generate.py photo.png --shape-only
"""

import argparse
import os
import sys
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Shim for basicsr/RealESRGAN compatibility with newer torchvision
import types
import torchvision.transforms.functional as _F
_mod = types.ModuleType("torchvision.transforms.functional_tensor")
_mod.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _mod

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")


def detect_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a textured 3D model from a single image.",
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", default="output.glb", help="Output GLB path (default: output.glb)")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (default: 50)")
    parser.add_argument("--octree-resolution", type=int, default=384, help="Shape detail level (default: 384)")
    parser.add_argument("--views", type=int, default=6, help="Number of texture views (default: 6)")
    parser.add_argument("--resolution", type=int, default=512, help="Texture resolution (default: 512)")
    parser.add_argument("--shape-only", action="store_true", help="Skip texture generation")
    parser.add_argument("--device", default=None, help="Force device (default: auto-detect mps/cuda/cpu)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: input image not found: {args.image}")
        sys.exit(1)

    device = args.device or detect_device()
    output_stem = os.path.splitext(args.output)[0]
    shape_path = f"{output_stem}_shape.glb"

    print(f"Input:  {args.image}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")

    t_start = time.time()

    # --- Step 1: Background Removal + Shape Generation ---
    print("\n=== Step 1: Shape Generation ===")
    from PIL import Image
    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    image = Image.open(args.image).convert("RGBA")
    rembg = BackgroundRemover()
    image = rembg(image)
    print(f"Image loaded and background removed: {image.size}")

    print("Loading shape model...")
    t0 = time.time()
    pipeline_shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        device=device,
    )
    print(f"Shape model loaded in {time.time() - t0:.1f}s")

    print(f"Generating shape ({args.steps} steps, octree {args.octree_resolution})...")
    t0 = time.time()
    mesh = pipeline_shape(
        image=image,
        num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution,
    )[0]
    print(f"Shape generated in {time.time() - t0:.1f}s")

    mesh.export(shape_path)
    shape_kb = os.path.getsize(shape_path) / 1024
    print(f"Shape saved: {shape_path} ({shape_kb:.0f} KB)")

    if args.shape_only:
        elapsed = time.time() - t_start
        print(f"\nDone (shape only) in {elapsed:.1f}s")
        return

    # --- Step 2: Texture Generation ---
    print("\n=== Step 2: Texture Generation ===")
    try:
        import tempfile
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        # Texture pipeline expects OBJ input
        obj_path = os.path.join(tempfile.gettempdir(), "hunyuan_shape.obj")
        mesh.export(obj_path)

        print(f"Configuring texture pipeline ({args.views} views, {args.resolution}px)...")
        conf = Hunyuan3DPaintConfig(max_num_view=args.views, resolution=args.resolution)
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"

        print("Loading texture models...")
        t0 = time.time()
        pipeline_tex = Hunyuan3DPaintPipeline(conf)
        print(f"Texture models loaded in {time.time() - t0:.1f}s")

        print("Generating texture...")
        t0 = time.time()
        pipeline_tex(
            mesh_path=obj_path,
            image_path=args.image,
            output_mesh_path=args.output,
            save_glb=True,
        )
        print(f"Texture generated in {time.time() - t0:.1f}s")

        if os.path.exists(args.output):
            size_kb = os.path.getsize(args.output) / 1024
            print(f"Textured output: {args.output} ({size_kb:.0f} KB)")
        else:
            print(f"WARNING: Output file not created. Shape-only mesh: {shape_path}")

    except Exception as e:
        print(f"\nTexture generation failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nShape-only mesh is still available at: {shape_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
