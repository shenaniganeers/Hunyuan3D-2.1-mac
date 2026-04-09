# Hunyuan3D-2.1 on macOS (Apple Silicon)

Generate textured 3D models from a single image on your Mac, using Metal GPU acceleration.

## Prerequisites

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4) — Intel Macs work but are much slower
- Python 3.11 (3.13 is not supported by PyTorch)
- 16 GB unified memory minimum (32 GB+ recommended for texture generation)
- ~15 GB disk space for models (downloaded on first run)

## Installation

```bash
git clone https://github.com/shenaniganeers/Hunyuan3D-2.1-mac.git
cd Hunyuan3D-2.1-mac

python3.11 -m venv .venv
source .venv/bin/activate

# PyTorch for macOS
pip install torch torchvision torchaudio

# Project dependencies
pip install -r requirements-macos.txt

# Build the Metal-accelerated rasterizer
cd hy3dpaint/custom_rasterizer
python setup.py build_ext --inplace
cd ../..

# Download the super-resolution model (used during texturing)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -P hy3dpaint/ckpt
```

## Environment

Always set this before running — some PyTorch MPS ops fall back to CPU:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Running

### Web UI (easiest)

```bash
python gradio_app.py --device mps
```

Add `--low_vram_mode` if you have less than 32 GB of unified memory.

Open http://127.0.0.1:8081, upload an image, click "Gen Shape", then "Gen Texture".

### Command line

**Shape only:**
```bash
python generate.py assets/demo.png -o model.glb --shape-only
```

**Full pipeline (shape + texture):**
```bash
python generate.py assets/demo.png -o model.glb
```

**All options:**
```bash
python generate.py photo.png -o model.glb \
  --steps 50 \
  --octree-resolution 384 \
  --views 6 \
  --resolution 512 \
  --device mps
```

### Python API

```python
import sys
sys.path.insert(0, './hy3dshape')

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh = pipeline(image='photo.png')[0]
mesh.export('shape.glb')
```

## How the pipeline works

```
Input image
  --> background removal (rembg)
  --> shape generation (3.3B DiT diffusion model, ~50 steps)
  --> 3D mesh (vertices + faces)
  --> multi-view rendering (6 camera angles, Metal rasterizer)
  --> texture synthesis (2B PBR diffusion model)
  --> super-resolution (RealESRGAN)
  --> UV baking
  --> textured GLB with PBR materials (albedo, metallic, roughness, normal)
```

## Models

All models download automatically from HuggingFace on first run and cache to `~/.cache/hy3dgen/`.

| Model | HuggingFace ID | Size |
|-------|---------------|------|
| Shape generator | `tencent/Hunyuan3D-2.1` / `hunyuan3d-dit-v2-1` | ~6 GB |
| Texture generator | `tencent/Hunyuan3D-2.1` / `hunyuan3d-paint-v2-1` | ~4 GB |
| Image encoder | `facebook/dinov2-giant` | ~2 GB |
| Super-resolution | `RealESRGAN_x4plus.pth` (manual download above) | 64 MB |

## Performance

| Stage | Memory | Time (Apple Silicon) |
|-------|--------|---------------------|
| Shape generation (50 steps) | 4-6 GB | 2-5 min |
| Texture generation (6 views) | 8-12 GB | 15-30 min |
| **Total** | **~12-18 GB peak** | **~20-35 min** |

With 64 GB unified memory you can run at full quality. Use `--low_vram_mode` only on machines with 16 GB or less.

For comparison, this runs 5-10x slower than on an NVIDIA GPU with CUDA.

## Metal rasterizer

The custom rasterizer has been ported from CUDA to Metal. It accelerates the multi-view rendering step of texture generation using GPU compute shaders.

**Build:**
```bash
cd hy3dpaint/custom_rasterizer
python setup.py build_ext --inplace
```

This compiles Metal shaders (`.metal` -> `.metallib`) and builds a Python C extension linked to Metal.framework. The extension is loaded as `custom_rasterizer_kernel` and must be imported after `torch` (for rpath resolution).

**Verify:**
```bash
cd hy3dpaint/custom_rasterizer
python -m pytest test_metal_rasterizer.py -v
```

If the Metal rasterizer is not built, the pipeline falls back to a pure-Python CPU rasterizer which is significantly slower.

## Troubleshooting

**"No module named setuptools" during build**
```bash
pip install setuptools
```

**MPS fallback warnings during inference**
Normal. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to suppress errors. Some operations don't have MPS implementations yet and run on CPU automatically.

**Out of memory**
- Use `--low_vram_mode`
- Close other apps to free unified memory
- Reduce inference steps (e.g., 20 instead of 50)
- Skip texture generation (`--disable_tex`)

**Model download fails**
Models cache to `~/.cache/hy3dgen/`. You can override with `HY3DGEN_MODELS=/path/to/models`. If download hangs, try downloading directly from https://huggingface.co/tencent/Hunyuan3D-2.1.

## Key files

```
generate.py                    # CLI: image → textured 3D model
gradio_app.py                  # Web interface
platform_utils.py              # Device detection (CUDA -> MPS -> CPU)
requirements-macos.txt         # macOS dependencies
hy3dshape/                     # Shape generation (DiT model)
hy3dpaint/                     # Texture generation (PBR model)
  custom_rasterizer/           # Metal GPU rasterizer
    setup.py                   # Builds Metal extension
    lib/custom_rasterizer_kernel/
      rasterizer_metal.metal   # Metal compute shaders
      rasterizer_metal.mm      # Obj-C++ bridge
      rasterizer.cpp           # C++ dispatcher
    test_metal_rasterizer.py   # Unit tests
```

## Links

- [Original repository (Tencent)](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [Models on HuggingFace](https://huggingface.co/tencent/Hunyuan3D-2.1)
- [Technical report](https://arxiv.org/abs/2506.15442)

## License

This project follows the original Hunyuan3D-2.1 license terms. See `LICENSE` for details.
