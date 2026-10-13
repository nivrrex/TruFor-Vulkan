# TruFor Operator Inventory (Phase 2)

- TruFor commit: see `git -C truFor-src rev-parse HEAD`
- Model: `weights/trufor.pth.tar` (experiment `trufor_ph3`)
- torch: `2.7.1+cpu`
- Runtime census image: `truFor-src/test_docker/images/tampered2.png` [1, 3, 256, 256]
- ExecuTorch Vulkan registry: `vulkan-op-support-table.csv` (112 operators)

## Summary

- **image**: truFor-src/test_docker/images/tampered2.png
- **runtime_total_aten_calls**: 2920
- **distinct_ops**: 41
- **supported_op_count**: 24
- **unsupported_op_count**: 17
- **aten_call_coverage_pct**: 37.26
- **export_ok**: True
- **export_error_type**: None
- **export_node_count**: 2826
- **registry_size**: 112

## Operators actually executed (sorted by call count)

| target | calls | in export graph | Vulkan | note |
|---|---:|:---:|---|---|
| `aten::view` | 805 | n | NOT_LISTED |  |
| `aten::expand` | 248 | n | NOT_LISTED |  |
| `aten::t` | 202 | n | NOT_LISTED |  |
| `aten::permute` | 200 | n | SUPPORTED |  |
| `aten::_unsafe_view` | 168 | n | NOT_LISTED |  |
| `aten::addmm` | 146 | n | SUPPORTED |  |
| `aten::add` | 142 | n | SUPPORTED |  |
| `aten::transpose` | 128 | n | NOT_LISTED |  |
| `aten::bmm` | 124 | n | SUPPORTED |  |
| `aten::native_layer_norm` | 114 | n | SUPPORTED | resize supported |
| `aten::convolution` | 111 | n | SUPPORTED | batch=1 for 2D conv; no transposed 1D conv; no 3D conv |
| `aten::clone` | 92 | n | SUPPORTED |  |
| `aten::select` | 80 | n | NOT_LISTED |  |
| `aten::mul` | 78 | n | NOT_LISTED |  |
| `aten::_softmax` | 40 | n | SUPPORTED |  |
| `aten::relu_` | 38 | n | NOT_LISTED |  |
| `aten::gelu` | 32 | n | SUPPORTED |  |
| `aten::cat` | 29 | n | SUPPORTED |  |
| `aten::empty` | 25 | n | NOT_LISTED |  |
| `aten::native_batch_norm` | 25 | n | NOT_LISTED |  |
| `aten::mm` | 12 | n | SUPPORTED |  |
| `aten::slice` | 10 | n | NOT_LISTED |  |
| `aten::sigmoid` | 8 | n | SUPPORTED |  |
| `aten::split` | 8 | n | SUPPORTED |  |
| `aten::unbind` | 8 | n | NOT_LISTED |  |
| `aten::upsample_bilinear2d` | 8 | n | SUPPORTED |  |
| `aten::unsqueeze` | 6 | n | NOT_LISTED |  |
| `aten::sub` | 4 | n | SUPPORTED |  |
| `aten::mean` | 4 | n | SUPPORTED | keepdim=True required; max 2D reductions |
| `aten::adaptive_max_pool2d` | 4 | n | NOT_LISTED |  |
| `aten::logsumexp` | 4 | n | NOT_LISTED |  |
| `aten::sum` | 4 | n | SUPPORTED | keepdim=True required; max 2D reductions |
| `aten::lift_fresh` | 2 | n | NOT_LISTED |  |
| `aten::_log_softmax` | 2 | n | SUPPORTED |  |
| `aten::neg` | 2 | n | SUPPORTED |  |
| `aten::exp` | 2 | n | SUPPORTED |  |
| `aten::repeat` | 1 | n | SUPPORTED |  |
| `aten::div` | 1 | n | SUPPORTED |  |
| `aten::zeros` | 1 | n | SUPPORTED |  |
| `aten::log_sigmoid_forward` | 1 | n | NOT_LISTED |  |
| `aten::relu` | 1 | n | SUPPORTED |  |

## Export graph

- full graph listing: `export-graph.txt`
- node count: 2826

## Operators with no Vulkan support (blocking candidates)

| target | calls | example args |
|---|---:|---|
| `aten::view` | 805 | `[['torch.float32', [1, 64, 64, 64]], '<list len=3>']` |
| `aten::expand` | 248 | `[['torch.float32', [1, 1, 4096, 64]], '<list len=4>']` |
| `aten::t` | 202 | `[['torch.float32', [64, 64]]]` |
| `aten::_unsafe_view` | 168 | `[['torch.float32', [1, 4096, 64]], '<list len=4>']` |
| `aten::transpose` | 128 | `[['torch.float32', [1, 64, 4096]], 1, 2]` |
| `aten::select` | 80 | `[['torch.float32', [2, 1, 1, 64, 64]], 0, 0]` |
| `aten::mul` | 78 | `[['torch.float32', [1, 1, 4096, 64]], 0.125]` |
| `aten::relu_` | 38 | `[['torch.float32', [1, 64, 256, 256]]]` |
| `aten::empty` | 25 | `['<list len=1>']` |
| `aten::native_batch_norm` | 25 | `[['torch.float32', [1, 64, 256, 256]], ['torch.float32', [64]], ['torch.float32', [64]], ['torch.float32', [64]], ['torch.float32', [64]], False, 0.1, 1e-05]` |
| `aten::slice` | 10 | `[['torch.float32', [1, 3]], 1, 0, 9223372036854775807]` |
| `aten::unbind` | 8 | `[['torch.float32', [2, 1, 1, 4096, 64]]]` |
| `aten::unsqueeze` | 6 | `[['torch.float32', [3]], 0]` |
| `aten::adaptive_max_pool2d` | 4 | `[['torch.float32', [1, 128, 64, 64]], '<list len=2>']` |
| `aten::logsumexp` | 4 | `[['torch.float32', [1, 1, 65536]], '<list len=1>']` |
| `aten::lift_fresh` | 2 | `[['torch.float32', [3]]]` |
| `aten::log_sigmoid_forward` | 1 | `[['torch.float32', [1, 1, 256, 256]]]` |
