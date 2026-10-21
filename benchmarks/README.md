# Benchmarks

How well do open attention models predict where real people look? `run.py` scores each model against human eye-tracking data with the standard saliency metrics, computed with [pysaliency](https://github.com/matthias-k/pysaliency) (the library behind the MIT/Tübingen benchmark).

```bash
pip install -e ".[unisal]" "git+https://github.com/matthias-k/pysaliency.git@dev" pandas scikit-image
pip install torch torchvision einops clip-anytorch "deepgaze_pytorch @ git+https://github.com/matthias-k/DeepGaze"  # DeepGaze IIE/MSDB
python benchmarks/run.py fiwi --models center,spectral,deepgaze_iie,deepgaze_msdb,transalnet,unisal
```

Datasets and third-party model repos live outside this repo, under `$MA_BENCH` (default `~/Documents/Personal/MediaAttention-bench`): `data/fiwi/`, `data/ueyes/`, `models/unisal/`, `models/TranSalNet/`. `run.py` explains where each comes from.

## Method

- **Ground truth:** each image's fixations, from all viewers, in image pixels. CC, KLD and SIM compare against those fixations blurred with the dataset's own Gaussian (FiWI: σ = 32 px).
- **Metrics** (Bylinskii et al. 2019), averaged per image:
  - **AUC:** all pixels count as negatives.
  - **sAUC:** other images' fixations count as negatives, which cancels centre bias.
  - **NSS**, **CC**, **SIM**: higher is better.
  - **KLD:** MIT/Tübingen regularisation; lower is better.
- **Models** run as shipped, with no fine-tuning on the test data:

  | Model | Weights and settings |
  |---|---|
  | DeepGaze IIE and MSDB | Gaussian centre bias; MSDB uses the dataset's pixels per degree |
  | TranSalNet-Dense | SALICON weights, its own resize and pad |
  | UNISAL | Base SALICON weights, through its own CLI |
  | Centre bias, spectral residual | Baselines |

- **Timing:** wall-clock per image on an Apple M3 Pro (MPS). It includes model loading only for UNISAL, which runs as a single batch.

## Caveats

- **FiWI is webpages, not ads.** It is small: 149 images, 11 viewers. Read it as indicative.
- **Contamination:** DeepGaze IIE was trained on MIT1003 and MSDB on several datasets, but none of these models saw FiWI.
- **Scores here are not comparable to MIT300 leaderboard numbers.** The dataset is different, and the leaderboard optimises each model's map per metric.
