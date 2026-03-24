# GCP Deployment Guide: Text-to-Image AI

Follow these step-by-step instructions to deploy your Text-to-Image application (with safety filter disabled) to a Google Cloud Platform (GCP) Compute Engine instance.

## 1. Provision a Compute Engine VM with GPU
To run Stable Diffusion efficiently, you need a GPU instance. The NVIDIA T4 GPU is a great, cost-effective choice on GCP.

1. Navigate to your GCP Console -> **Compute Engine** -> **VM instances** -> **Create Instance**.
2. **Name**: `stable-diffusion-server`
3. **Region/Zone**: Choose a zone that has T4 GPUs available (e.g., `us-central1-a`, `us-east1-d`, `europe-west4-a`).
4. **Machine configuration**:
   - Machine family: **GPU**
   - GPU type: **NVIDIA T4** (Count: 1)
   - Machine type: **n1-standard-4** (4 vCPUs, 15 GB memory)
5. **Boot disk**:
   - Click **Change**.
   - Operating System: **Deep Learning on Linux**
   - Version: **Deep Learning VM with CUDA 12.0 M113 Ubuntu 22.04** (or similar latest version). These images come with NVIDIA drivers and Docker pre-installed!
   - Boot disk type: **Balanced persistent disk** or **SSD**
   - Size: Allocate at least **60 GB**. The Docker image including PyTorch and the model weights will exceed 15-20 GB.
6. **Firewall**: 
   - Check **Allow HTTP traffic** and **Allow HTTPS traffic**.
7. **Advanced Options -> Networking**:
   - Add a network tag: `allow-gradio-7860`. We will use this to open port 7860 in the next step.
8. Click **Create** (Note: If you get a quota error on GCP, you may need to request a quota increase for "GPUs (all regions)" or "NVIDIA T4 GPUs" in the IAM & Admin -> Quotas page. GCP approvals are often faster than AWS).

## 2. Open Port `7860` in GCP Firewall
We need to allow traffic on port 7860 for Gradio.
1. Go to **VPC network** -> **Firewall**.
2. Click **Create Firewall Rule**.
3. **Name**: `allow-gradio-7860`
4. **Target tags**: `allow-gradio-7860`
5. **Source IPv4 ranges**: `0.0.0.0/0`
6. **Protocols and ports**: Select **Specified protocols and ports**, check **TCP**, and enter `7860`.
7. Click **Create**.

## 3. Connect to the Instance
1. Back in **Compute Engine -> VM instances**, find your `stable-diffusion-server`.
2. Click the **SSH** button next to your instance to open a browser-based terminal.
   *(Alternatively, use the `gcloud` CLI: `gcloud compute ssh stable-diffusion-server --zone=<your-zone>`)*

Since we selected the Deep Learning VM, **Docker and the NVIDIA Container Toolkit are already installed!** You may be prompted to install the NVIDIA driver on the first SSH connection; type `Y` and press Enter if prompted.

## 4. Upload Your Code to the Server
You can clone your git repo directly on the remote server, or upload files using the SSH terminal's "Upload file" button (top right gear icon in the browser SSH window).

Ensure that `requirements.txt`, `app.py`, and `Dockerfile` are placed in the same directory (e.g., `/home/<your-username>/`).

## 5. Build and Run the Docker Container
In the directory where your `Dockerfile` is located:

1. **Build the image**:
   *(Warning: This will take several minutes as it downloads PyTorch and the 4GB+ Stable Diffusion model weights into the Docker image.)*
   ```bash
   sudo docker build -t stable-diffusion-app .
   ```

2. **Run the container**:
   ```bash
   sudo docker run --gpus all -d -p 7860:7860 stable-diffusion-app
   ```
   *Note the `--gpus all` flag—this gives the container direct access to the NVIDIA T4 GPU.*

## 6. Access the Application
1. Find your VM's **External IP** in the GCP Console.
2. Open any web browser and navigate to:
   ```
   http://<your-gcp-external-ip>:7860
   ```
   *(Make sure you use `http://` and not `https://`)*

You should see your text-to-image application ready to use!
