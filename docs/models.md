# Depth models

shiftpaper works out the depth of each image with a depth model. That happens once per image, when it's baked by `shiftpaper set`, `slideshow` or `bake`. The daemon only reads the result, so it never runs the model itself.

## Getting a model

```
shiftpaper fetch-model                # the default, MoGe-2 ViT-B
shiftpaper fetch-model --list         # every model you can download, with sizes
shiftpaper fetch-model moge-2-vits    # a particular one
```

Whichever you fetch last is the one used from then on. The default gives the most detail but is the slowest to bake. If baking feels slow, try `moge-2-vits` or `depth-anything-v2-small`.

Each model has its own licence, shown before it downloads. They're all listed in [THIRD_PARTY.md](../THIRD_PARTY.md).

## Switching models

Wallpapers you've already set keep the depth map they were baked with. To bake one again with the new model, `shiftpaper set` it again.

You can also use any Depth Anything or MoGe-2 ONNX file you have, with `--model path/to/model.onnx` or by setting `SHIFTPAPER_MODEL`.

## Where things go

| What | Where |
|---|---|
| Downloaded models | `~/.local/share/shiftpaper/models/` |
| Baked wallpapers | `~/.cache/shiftpaper/wallpapers/` |

Each baked wallpaper is a pair of PNGs: the colour image and a 16-bit depth map. They're named after the image and the model, so baking the same image again just reuses them. You can clear out the cache whenever you like, but anything you've set from it will need setting again.

## Baking on a GPU

On an NVIDIA card, install `onnxruntime-cuda` (on Arch) and baking uses the GPU. Otherwise it runs on the CPU, which is slower but fine for a handful of images.
