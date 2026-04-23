# Convolution Triton Template

Canonical Triton skeleton for convolutions — `conv1d`, `conv2d`, `conv3d`, depthwise, and
grouped. Two approaches coexist: **(A) direct tiled convolution** (shown below) for small
kernels and non-unit strides; **(B) im2col + GEMM** for large kernels — in that case,
lower to `gemm.md` after an im2col transform. This skeleton shows (A) for conv2d.

```python
import triton
import triton.language as tl
import torch

# Direct conv: tile the output over (N, C_out, H_out, W_out). Each program
# computes a BLOCK_OH x BLOCK_OW output tile for one (batch, out-channel-group).
# Loop over (R, S, C_in) for the reduction.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 8,  'BLOCK_OW': 32, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_OH': 4,  'BLOCK_OW': 64, 'BLOCK_C': 32}, num_warps=8),
    ],
    key=['H_out', 'W_out', 'C_in', 'K_h', 'K_w'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, y_ptr,               # NCHW x, KCRS w, NKHW y
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w, stride_h, stride_w, pad_h, pad_w,
    # strides of x (NCHW) and y (NKHW) — pass from python
    sx_n, sx_c, sx_h, sx_w,
    sy_n, sy_k, sy_h, sy_w,
    sw_k, sw_c, sw_r, sw_s,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Program grid: (batch*out_channels, ceil(H_out/BLOCK_OH), ceil(W_out/BLOCK_OW))
    pid_nk = tl.program_id(axis=0)
    pid_h  = tl.program_id(axis=1)
    pid_w  = tl.program_id(axis=2)

    n = pid_nk // C_out
    k = pid_nk %  C_out

    offs_oh = pid_h * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = pid_w * BLOCK_OW + tl.arange(0, BLOCK_OW)
    mask_oh = offs_oh < H_out
    mask_ow = offs_ow < W_out

    # fp32 accumulator — conv K-loop is long (C_in * K_h * K_w), low precision drifts.
    acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    # Reduction loop over (C_in, K_h, K_w). Peel C_in into BLOCK_C tiles so
    # we can load a C_in chunk of x and w at once.
    for c_start in range(0, C_in, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_in
        for r in range(K_h):
            for s in range(K_w):
                # Input coords for this (r, s, c) filter position.
                # Each (oh, ow) maps to input (oh*stride_h + r - pad_h, ow*stride_w + s - pad_w).
                ih = offs_oh[:, None] * stride_h + r - pad_h     # [BLOCK_OH, 1]
                iw = offs_ow[None, :] * stride_w + s - pad_w     # [1, BLOCK_OW]
                in_bounds = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)

                # Load x[n, c, ih, iw] — gather, masked for bounds AND channel tail.
                x_off = (n * sx_n + offs_c[:, None, None] * sx_c
                         + ih[None, :, :] * sx_h + iw[None, :, :] * sx_w)
                x_tile = tl.load(
                    x_ptr + x_off,
                    mask=in_bounds[None, :, :] & mask_c[:, None, None],
                    other=0.0,
                )  # [BLOCK_C, BLOCK_OH, BLOCK_OW]

                # Load w[k, c, r, s] — one per c.
                w_off = k * sw_k + offs_c * sw_c + r * sw_r + s * sw_s
                w_tile = tl.load(w_ptr + w_off, mask=mask_c, other=0.0)  # [BLOCK_C]

                acc += tl.sum(x_tile * w_tile[:, None, None], axis=0)

    # --- EPILOGUE: bias, activation, batchnorm-in-inference, etc. ---
    # acc += tl.load(bias_ptr + k)
    # acc = tl.where(acc > 0, acc, 0.0)
    # ------------------------------------------------------------

    y_off = (n * sy_n + k * sy_k
             + offs_oh[:, None] * sy_h + offs_ow[None, :] * sy_w)
    tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty),
             mask=mask_oh[:, None] & mask_ow[None, :])


def conv2d_launch(x, w, stride=1, padding=0):
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = w.shape
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    H_out = (H_in + 2*ph - K_h) // sh + 1
    W_out = (W_in + 2*pw - K_w) // sw + 1
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    grid = lambda m: (N*C_out, triton.cdiv(H_out, m['BLOCK_OH']), triton.cdiv(W_out, m['BLOCK_OW']))
    conv2d_kernel[grid](
        x, w, y, N, C_in, H_in, W_in, C_out, H_out, W_out,
        K_h, K_w, sh, sw, ph, pw,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
    )
    return y
```

## Caveats

- **Direct vs im2col+GEMM.** Direct (above) wins for 1x1, 3x3, and small kernels — no
  memory blowup. For 7x7+ or large batched convs, im2col-into-GEMM wins because it
  hands work to the tuned GEMM kernel. Rule of thumb: `K_h * K_w <= 9` → direct.
- **Boundary masking is mandatory.** Padding means input indices can be negative or
  out-of-range; always compute `in_bounds` and mask the load. `other=0.0` implements
  zero padding implicitly.
- **fp32 accumulator.** C_in × K_h × K_w can be thousands; low-precision accumulation
  drifts. Promote at load or let `tl.dot` accumulate in fp32 if using im2col+GEMM.
- **Depthwise conv.** `C_in == C_out == groups`, each output channel reads exactly one
  input channel. Skip the C_in reduction dim entirely — one program per (n, c, oh tile).
- **Grouped conv.** Split C_in and C_out into `groups`; each program handles one group.
  Pass `group_id` via program_id axis, offset channel indices accordingly.
- **1D / 3D conv.** Same skeleton, reduce/add spatial dims. 1D: drop `K_w`, `W`,
  BLOCK_OW. 3D: add `K_d`, `D`, BLOCK_OD.
- **Stride / dilation.** `stride_h/w` already shown. Dilation: multiply `r`, `s` by
  dilation factor when computing `ih`, `iw`.
- **When NOT to use.** For plain `conv2d 1x1`, this is just a GEMM — use `gemm.md`
  after reshaping (N, C_in, H, W) → (N*H*W, C_in) × (C_in, C_out).
