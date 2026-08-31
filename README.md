# CVINet — Cognitive Visual Imagination for Camouflaged Object Detection

Official implementation of [*Seeing the unseen: Cognitive Visual Imagination for
camouflaged object detection*](https://doi.org/10.1016/j.knosys.2026.116974),
Knowledge-Based Systems, 2026.

A frozen image-editing model first turns each input into a **mental template** that
reveals the target and suppresses the background. CVINet then segments the image and
its template as two streams, coupled by **CSI** (semantic injection in the encoder)
and **CSR** (high-frequency structure refinement in the decoder).

## Download

- [CVINet weights](https://github.com/suhang2000/CVINet/releases/download/v1.0/CVINet_pvtv2b4.pt) — PVTv2-B4 backbone
- [Prediction maps](https://github.com/suhang2000/CVINet/releases/download/v1.0/CVINet_prediction_maps.zip) — all four benchmarks

## Results

| Dataset | S<sub>&alpha;</sub> | E<sub>&phi;</sub> | F<sub>&beta;</sub><sup>w</sup> | MAE |
|---|---|---|---|---|
| CHAMELEON | 0.923 | 0.964 | 0.892 | 0.021 |
| CAMO | 0.885 | 0.936 | 0.851 | 0.042 |
| COD10K | 0.882 | 0.941 | 0.818 | 0.021 |
| NC4K | 0.897 | 0.942 | 0.859 | 0.029 |

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.6.0, CUDA 12.4. Stage 1 additionally needs
`diffusers`, `transformers` and `accelerate` — see `requirements.txt`.

## Data

[COD benchmarks](https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod).
Images, templates and masks are matched by filename:

```
data/{train,test}/
├── image/       # original camouflaged images
├── template/    # generated mental templates
└── gt/          # ground-truth masks
```

## Usage

Stage 1 — generate templates. Uses the input image and a fixed instruction prompt
only; no ground truth. Deterministic per-image seeds, and reruns skip existing files.

```bash
python gen_image.py --data-dir ./data/train/image \
                    --output-dir ./data/train/template \
                    --model-id Qwen/Qwen-Image-Edit-2509
```

Stage 2 — train, then test.

```bash
python train.py --image-dir ./data/train/image --template-dir ./data/train/template \
                --gt-dir ./data/train/gt --output-dir ./checkpoints

python test.py  --image-dir ./data/test/image  --template-dir ./data/test/template \
                --gt-dir ./data/test/gt --checkpoint ./checkpoints/epoch_50.pt \
                --output-dir ./test_outputs
```

Drop `--gt-dir` from `test.py` to predict without evaluating. Metrics are computed at
the 416 input resolution; predicted masks are written at the original image resolution.

## Citation

```bibtex
@article{li2026cvicod,
  title   = {Seeing the unseen: Cognitive Visual Imagination for camouflaged object detection},
  author  = {Li, Suhang and Yoshie, Osamu and Ieiri, Yuya},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  pages   = {116974},
  doi     = {10.1016/j.knosys.2026.116974}
}
```

## License

[MIT](LICENSE).
