# Third-party models

shiftpaper doesn't include any model weights. `shiftpaper fetch-model` downloads them from HuggingFace, and each one is licensed by its authors, not under shiftpaper's MIT licence.

`fetch-model` only lists models with permissive licences. Anything you use with `--model` comes with its own terms.

| Model | Weights licence | Authors | Upstream | ONNX export |
|---|---|---|---|---|
| `moge-2-vitb` (default) | MIT | Microsoft Research | [Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) | [Ruicheng/moge-2-vitb-normal-onnx](https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx) |
| `moge-2-vits` | MIT | Microsoft Research | [Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal) | [Ruicheng/moge-2-vits-normal-onnx](https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx) |
| `depth-anything-v3-base` | Apache-2.0 | ByteDance Seed | [depth-anything/DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | [onnx-community/depth-anything-v3-base](https://huggingface.co/onnx-community/depth-anything-v3-base) |
| `depth-anything-v3-small` | Apache-2.0 | ByteDance Seed | [depth-anything/DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | [onnx-community/depth-anything-v3-small](https://huggingface.co/onnx-community/depth-anything-v3-small) |
| `depth-anything-v2-small` | Apache-2.0 | HKU and TikTok | [depth-anything/Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) | [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small) |

Licences are as stated on each upstream model card, which is where to check if in doubt.
