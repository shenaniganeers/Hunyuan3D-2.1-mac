#include "rasterizer_metal.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <string>

// Cached Metal state (initialized once per process)
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static id<MTLComputePipelineState> g_rasterizePipeline = nil;
static id<MTLComputePipelineState> g_barycentricPipeline = nil;
static bool g_initialized = false;

static void ensureMetalInitialized() {
    if (g_initialized) return;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(g_device != nil, "Metal: No GPU device found");

        g_commandQueue = [g_device newCommandQueue];
        TORCH_CHECK(g_commandQueue != nil, "Metal: Failed to create command queue");

        // Locate the .metallib next to this compiled library
        NSString* libPath = nil;
        const char* soPathCStr = nullptr;

        Dl_info info;
        if (dladdr((void*)&ensureMetalInitialized, &info) && info.dli_fname) {
            soPathCStr = info.dli_fname;
            NSString* soPath = [NSString stringWithUTF8String:info.dli_fname];
            NSString* soDir = [soPath stringByDeletingLastPathComponent];

            // Search several possible locations for the metallib
            NSArray* candidates = @[
                [soDir stringByAppendingPathComponent:@"rasterizer.metallib"],
                [[soDir stringByAppendingPathComponent:@"../lib"]
                    stringByAppendingPathComponent:@"rasterizer.metallib"],
                [[soDir stringByAppendingPathComponent:@"lib"]
                    stringByAppendingPathComponent:@"rasterizer.metallib"],
                [[soDir stringByAppendingPathComponent:@"../lib/custom_rasterizer_kernel"]
                    stringByAppendingPathComponent:@"rasterizer.metallib"],
                [soDir stringByAppendingPathComponent:@"lib/custom_rasterizer_kernel/rasterizer.metallib"],
            ];

            for (NSString* candidate in candidates) {
                if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
                    libPath = candidate;
                    break;
                }
            }
        }

        TORCH_CHECK(libPath != nil,
            "Metal: Could not find rasterizer.metallib. Searched near: ",
            soPathCStr ? soPathCStr : "(unknown)");

        NSError* error = nil;
        NSURL* libURL = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> library = [g_device newLibraryWithURL:libURL error:&error];
        TORCH_CHECK(library != nil, "Metal: Failed to load metallib: ",
            error ? [[error localizedDescription] UTF8String] : "unknown error");

        // Create pipeline for rasterizeImagecoords kernel
        id<MTLFunction> rasterizeFunc = [library newFunctionWithName:@"rasterizeImagecoords"];
        TORCH_CHECK(rasterizeFunc != nil, "Metal: Could not find 'rasterizeImagecoords' kernel");
        g_rasterizePipeline = [g_device newComputePipelineStateWithFunction:rasterizeFunc error:&error];
        TORCH_CHECK(g_rasterizePipeline != nil, "Metal: Failed to create rasterize pipeline: ",
            error ? [[error localizedDescription] UTF8String] : "unknown error");

        // Create pipeline for barycentricFromImgcoord kernel
        id<MTLFunction> barycentricFunc = [library newFunctionWithName:@"barycentricFromImgcoord"];
        TORCH_CHECK(barycentricFunc != nil, "Metal: Could not find 'barycentricFromImgcoord' kernel");
        g_barycentricPipeline = [g_device newComputePipelineStateWithFunction:barycentricFunc error:&error];
        TORCH_CHECK(g_barycentricPipeline != nil, "Metal: Failed to create barycentric pipeline: ",
            error ? [[error localizedDescription] UTF8String] : "unknown error");

        g_initialized = true;
    }
}

// Helper: create a Metal buffer wrapping tensor data.
// newBufferWithBytesNoCopy requires page-aligned pointers;
// if alignment fails, fall back to a copy.
static id<MTLBuffer> tensorToMTLBuffer(id<MTLDevice> device, torch::Tensor& t) {
    void* ptr = t.data_ptr();
    size_t size = t.nbytes();

    // Try zero-copy first (requires page-aligned pointer)
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:ptr
                         length:size
                         options:MTLResourceStorageModeShared
                         deallocator:nil];
    if (buf != nil) return buf;

    // Fall back to copy
    buf = [device newBufferWithBytes:ptr
            length:size
            options:MTLResourceStorageModeShared];
    return buf;
}

// Helper: copy Metal buffer contents back to tensor (for fallback copy path)
static void copyMTLBufferToTensor(id<MTLBuffer> buf, torch::Tensor& t) {
    memcpy(t.data_ptr(), [buf contents], t.nbytes());
}

std::vector<torch::Tensor> rasterize_image_metal(
    torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior)
{
    @autoreleasepool {
        ensureMetalInitialized();

        // Ensure tensors are contiguous and on CPU with correct types
        V = V.contiguous().to(torch::kCPU, torch::kFloat32);
        F = F.contiguous().to(torch::kCPU, torch::kInt32);
        if (use_depth_prior && D.numel() > 0) {
            D = D.contiguous().to(torch::kCPU, torch::kFloat32);
        }

        uint32_t num_faces = (uint32_t)F.size(0);
        uint32_t num_vertices = (uint32_t)V.size(0);
        uint32_t w = (uint32_t)width;
        uint32_t h = (uint32_t)height;
        uint32_t use_dp = (uint32_t)use_depth_prior;

        // Allocate output tensors
        auto int_options = torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
        auto float_options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

        auto findices = torch::zeros({height, width}, int_options);
        auto barycentric = torch::zeros({height, width, 3}, float_options);

        // Early return for empty mesh — no faces to rasterize
        if (num_faces == 0) {
            return {findices, barycentric};
        }

        // Z-buffer: two uint32 buffers
        // zbuf_depth: initialized to 0xFFFFFFFF (max distance)
        // zbuf_faceId: initialized to 0 (no face)
        auto zbuf_depth = torch::full({height, width}, (int32_t)0xFFFFFFFF, int_options);
        auto zbuf_faceId = torch::zeros({height, width}, int_options);

        // Create Metal buffers
        id<MTLBuffer> V_buf = tensorToMTLBuffer(g_device, V);
        id<MTLBuffer> F_buf = tensorToMTLBuffer(g_device, F);
        id<MTLBuffer> zbuf_depth_buf = tensorToMTLBuffer(g_device, zbuf_depth);
        id<MTLBuffer> zbuf_faceId_buf = tensorToMTLBuffer(g_device, zbuf_faceId);
        id<MTLBuffer> findices_buf = tensorToMTLBuffer(g_device, findices);
        id<MTLBuffer> bary_buf = tensorToMTLBuffer(g_device, barycentric);

        // Depth prior buffer
        id<MTLBuffer> D_buf = nil;
        if (use_depth_prior && D.numel() > 0) {
            D_buf = tensorToMTLBuffer(g_device, D);
        } else {
            // Metal requires a valid buffer for all bound indices
            float dummy = 0.0f;
            D_buf = [g_device newBufferWithBytes:&dummy length:sizeof(float)
                     options:MTLResourceStorageModeShared];
        }

        TORCH_CHECK(V_buf && F_buf && zbuf_depth_buf && zbuf_faceId_buf &&
                     findices_buf && bary_buf && D_buf,
            "Metal: Failed to create one or more Metal buffers");

        // --- Kernel 1: rasterizeImagecoords ---
        id<MTLCommandBuffer> cmdBuffer = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_rasterizePipeline];
        [encoder setBuffer:V_buf offset:0 atIndex:0];
        [encoder setBuffer:F_buf offset:0 atIndex:1];
        [encoder setBuffer:D_buf offset:0 atIndex:2];
        [encoder setBuffer:zbuf_depth_buf offset:0 atIndex:3];
        [encoder setBuffer:zbuf_faceId_buf offset:0 atIndex:4];
        [encoder setBytes:&w length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&h length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&num_faces length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&occlusion_truncation length:sizeof(float) atIndex:8];
        [encoder setBytes:&use_dp length:sizeof(uint32_t) atIndex:9];

        NSUInteger threadsPerGroup1 = MIN(g_rasterizePipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize gridSize1 = MTLSizeMake(num_faces, 1, 1);
        MTLSize groupSize1 = MTLSizeMake(threadsPerGroup1, 1, 1);
        [encoder dispatchThreads:gridSize1 threadsPerThreadgroup:groupSize1];

        [encoder endEncoding];

        // --- Kernel 2: barycentricFromImgcoord ---
        // Separate encoder ensures kernel 1 completes before kernel 2 reads buffers
        id<MTLComputeCommandEncoder> encoder2 = [cmdBuffer computeCommandEncoder];

        [encoder2 setComputePipelineState:g_barycentricPipeline];
        [encoder2 setBuffer:V_buf offset:0 atIndex:0];
        [encoder2 setBuffer:F_buf offset:0 atIndex:1];
        [encoder2 setBuffer:findices_buf offset:0 atIndex:2];
        [encoder2 setBuffer:zbuf_faceId_buf offset:0 atIndex:3];
        [encoder2 setBuffer:bary_buf offset:0 atIndex:4];
        [encoder2 setBytes:&w length:sizeof(uint32_t) atIndex:5];
        [encoder2 setBytes:&h length:sizeof(uint32_t) atIndex:6];
        [encoder2 setBytes:&num_vertices length:sizeof(uint32_t) atIndex:7];
        [encoder2 setBytes:&num_faces length:sizeof(uint32_t) atIndex:8];

        NSUInteger threadsPerGroup2 = MIN(g_barycentricPipeline.maxTotalThreadsPerThreadgroup, 256);
        NSUInteger totalPixels = (NSUInteger)(width * height);
        MTLSize gridSize2 = MTLSizeMake(totalPixels, 1, 1);
        MTLSize groupSize2 = MTLSizeMake(threadsPerGroup2, 1, 1);
        [encoder2 dispatchThreads:gridSize2 threadsPerThreadgroup:groupSize2];

        [encoder2 endEncoding];

        // Submit and wait
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (cmdBuffer.error) {
            TORCH_CHECK(false, "Metal: Command buffer execution failed: ",
                [[cmdBuffer.error localizedDescription] UTF8String]);
        }

        // If we used copy-based buffers, copy results back
        // (tensorToMTLBuffer may have copied if pointer wasn't page-aligned)
        if ([findices_buf contents] != findices.data_ptr()) {
            copyMTLBufferToTensor(findices_buf, findices);
        }
        if ([bary_buf contents] != barycentric.data_ptr()) {
            copyMTLBufferToTensor(bary_buf, barycentric);
        }

        return {findices, barycentric};
    }
}
