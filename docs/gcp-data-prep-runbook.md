# GCP Data Prep Runbook

One-time job to pre-fetch all training data to GCS. Run before any real training run.

## Prerequisites

- `gcloud` CLI authenticated (`gcloud auth login` or service account)
- `GOOGLE_APPLICATION_CREDENTIALS` pointing to the service account key
- `MHW_GCS_BUCKET` set to your GCS bucket URI (see `mondal-mhw-gcp-info.md`)
- GEE whitelist approved for WeatherNext 2 (already done — see recent actions log)

## 1. Create the spot VM

```bash
gcloud compute instances create mhw-data-prep \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-standard \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=cloud-platform
```

## 2. SSH into the VM

```bash
gcloud compute ssh mhw-data-prep --zone=us-central1-a
```

## 3. Set up the environment on the VM

```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh

# Clone repo and create env
git clone <your-repo-url> mhw-risk-profiler
cd mhw-risk-profiler
conda env create -f environment.yml  # or: conda create -n mhw-risk python=3.11 && pip install -r requirements.txt

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
export MHW_GCS_BUCKET=gs://your-bucket-name   # see mondal-mhw-gcp-info.md

# Copy credentials to VM (from local machine, in a separate terminal):
# gcloud compute scp ~/.config/gcp-keys/mhw-harvester.json mhw-data-prep:~/.config/gcp-keys/ --zone=us-central1-a
```

## 4. Run the data prep job

```bash
conda run -n mhw-risk python scripts/run_data_prep.py 2>&1 | tee data_prep.log
```

Estimated runtime: 3-5 hours. The job is idempotent — if the VM is preempted, re-run the
same command and it will skip completed steps.

## 5. Verify outputs

```bash
conda run -n mhw-risk python -c "
import os, xarray as xr
b = os.environ['MHW_GCS_BUCKET']
for path in [
    f'{b}/hycom/tiles/2022/', f'{b}/hycom/tiles/2023/',
    f'{b}/hycom/climatology/',
    f'{b}/era5/2022/', f'{b}/era5/2023/',
]:
    ds = xr.open_zarr(path)
    print(path, '->', dict(ds.sizes))
"
```

Expected: each path prints non-empty dims.

## 6. Delete the VM

```bash
gcloud compute instances delete mhw-data-prep --zone=us-central1-a
```

## 7. Run real training (from local machine or separate GCE GPU VM)

```bash
export MHW_GCS_BUCKET=gs://your-bucket-name
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp-keys/mhw-harvester.json
conda run -n mhw-risk python scripts/train_era5.py --epochs 50
conda run -n mhw-risk python scripts/train_wn2.py --epochs 50
```
