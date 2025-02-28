#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "ext_qmatrix.h"

#include "cuda/q_matrix.cuh"
#include "cuda/q_gemm.cuh"

#include "cpp/util.h"


uintptr_t make_q_matrix
(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor q_scale,
    torch::Tensor q_scale_max,
    torch::Tensor q_groups,
    torch::Tensor q_group_map,
    torch::Tensor gptq_qzeros,
    torch::Tensor gptq_scales,
    torch::Tensor gptq_g_idx,
    torch::Tensor bias,
    torch::Tensor temp_dq
)
{
    TORCH_CHECK_DTYPE(q_weight, kInt);
    TORCH_CHECK_DTYPE_OPT(q_perm, kShort);
    TORCH_CHECK_DTYPE_OPT(q_invperm, kShort);
    TORCH_CHECK_DTYPE_OPT(q_scale, kInt);
    TORCH_CHECK_DTYPE_OPT(q_scale_max, kHalf);
    TORCH_CHECK_DTYPE_OPT(q_groups, kShort);
    TORCH_CHECK_DTYPE_OPT(q_group_map, kShort);
    TORCH_CHECK_DTYPE_OPT(gptq_qzeros, kInt);
    TORCH_CHECK_DTYPE_OPT(gptq_scales, kHalf);
    TORCH_CHECK_DTYPE_OPT(gptq_g_idx, kInt);
    TORCH_CHECK_DTYPE_OPT(bias, kHalf);

    TORCH_CHECK_SHAPES(q_perm, 0, q_invperm, 0, 1);

    int device = q_weight.device().index();
    int width = q_weight.size(1);
    int groups;
    int height;

    if (!q_scale.device().is_meta())
    {
        TORCH_CHECK_SHAPES(q_weight, 1, q_scale, 1, 8);
        TORCH_CHECK_SHAPES(q_scale_max, 0, q_scale, 0, 1);
        groups = q_scale.size(0);
        height = q_invperm.size(0);
    }
    else
    {
        TORCH_CHECK_SHAPES(q_weight, 1, gptq_qzeros, 1, 8);
        TORCH_CHECK_SHAPES(q_weight, 1, gptq_scales, 1, 1);
        groups = gptq_qzeros.size(0);
        height = q_weight.size(0) * 8;
    }

    if (!bias.device().is_meta())
    {
        TORCH_CHECK_SHAPES(q_weight, 1, bias, 0, 1);
    }

    TORCH_CHECK(temp_dq.size(0) >= width * height, "Insufficient size of temp_dq buffer")

    QMatrix* m = new QMatrix
    (
        device,
        height,
        width,
        groups,
        (uint32_t*) q_weight.data_ptr(),
        q_perm.device().is_meta() ? NULL : (uint16_t*) q_perm.data_ptr(),
        q_invperm.device().is_meta() ? NULL : (uint16_t*) q_invperm.data_ptr(),
        q_scale.device().is_meta() ? NULL : (uint32_t*) q_scale.data_ptr(),
        q_scale_max.device().is_meta() ? NULL : (half*) q_scale_max.data_ptr(),
        q_groups.device().is_meta() ? NULL : (uint16_t*) q_groups.data_ptr(),
        q_group_map.device().is_meta() ? NULL : (uint16_t*) q_group_map.data_ptr(),
        gptq_qzeros.device().is_meta() ? NULL : (uint32_t*) gptq_qzeros.data_ptr(),
        gptq_scales.device().is_meta() ? NULL : (half*) gptq_scales.data_ptr(),
        gptq_g_idx.device().is_meta() ? NULL : (uint32_t*) gptq_g_idx.data_ptr(),
        bias.device().is_meta() ? NULL : (half*) bias.data_ptr(),
        (half*) temp_dq.data_ptr()
    );

    if (m->failed) throw std::runtime_error("CUDA out of memory");

    return reinterpret_cast<uintptr_t> (m);
}

void free_q_matrix
(
    uintptr_t handle
)
{
    QMatrix* m = reinterpret_cast<QMatrix*> (handle);
    delete m;
}

void reconstruct
(
    uintptr_t q_handle,
    torch::Tensor output
)
{
    QMatrix* qm = reinterpret_cast<QMatrix*> (q_handle);
    TORCH_CHECK(qm->height == output.size(0) && qm->width == output.size(1), "Output tensor doesn't match shape of QMatrix")
    TORCH_CHECK_DTYPE(output, kHalf);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));

    qm->reconstruct((half*) output.data_ptr());
}


void gemm_half_q_half
(
    torch::Tensor a,
    uintptr_t b,
    torch::Tensor c,
    bool force_cuda
)
{
    QMatrix* qm = reinterpret_cast<QMatrix*> (b);

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(c, kHalf);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK(qm->height == a.size(1), "a and b have incompatible shapes")
    TORCH_CHECK(qm->width == c.size(1), "b and c have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

    gemm_half_q_half_cuda
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        qm,
        (half*) c.data_ptr(),
        c.size(0), // m
        c.size(1), // n
        a.size(1), // k
        true,
        NULL,
        force_cuda
    );
}
