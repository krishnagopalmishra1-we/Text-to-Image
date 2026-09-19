# GCP GPU Quota Request Guide for Flux.1 [dev]

To deploy a large text-to-image model like **Flux.1 [dev]** (a 12B parameter model), you need a powerful GPU with sufficient VRAM (Video RAM). 

## 1. Determine GPU Requirements for Flux.1 [dev]
* **NVIDIA A100 (40GB or 80GB)**: Highly recommended for full `bfloat16` inference. It fits the model and provides excellent generation speed.
* **NVIDIA L4 (24GB)**: Good cost-effective target. It can run the model with optimization (like 8-bit quantization or sequential CPU offloading), though it will be slightly slower.
* **NVIDIA T4 (16GB)**: Not recommended. Too little VRAM for Flux.1 [dev] unless heavily quantized (e.g., GGUF NF4), which drastically reduces performance.

**Recommendation**: Request quota for **NVIDIA L4** or **NVIDIA A100** GPUs.

## 2. Prerequisites
1. **Upgraded Billing Account**: You **cannot** request GPU quotas on a Google Cloud Free Trial account. You must upgrade to a paid account (Navigate to **Billing** -> **Upgrade**).
2. **Project Setup**: Ensure your Google Cloud Project is selected in the top dropdown.

## 3. Steps to Request Quota

### Step 3.1: Navigate to Quotas Page
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. In the top search bar, type **Quotas** and click on **Quotas & System Limits** (under IAM & Admin).

### Step 3.2: Find the Global GPU Quota
*GCP restricts both the total number of GPUs globally and the specific type of GPU in a region.*
1. In the **Filter** box, enter: `Quota: GPUs (all regions)`
2. Check the box next to **GPUs (all regions)**.

### Step 3.3: Find the Specific GPU Model Quota
1. In the **Filter** box, enter your desired GPU, for example: `Quota: NVIDIA L4 GPUs` (or `NVIDIA A100 GPUs`).
2. Add another filter for **Region**. Choose a region close to you that has GPU availability.
    * Example: `Region: us-central1`
    * *Tip: `us-central1`, `us-east4`, or `europe-west4` usually have good availability, but you should pick what's closest.*
3. Check the box next to the specific GPU in your chosen region.

### Step 3.4: Submit the Request
1. With both checkboxes selected (Global GPUs and Specific Region GPU), click the **EDIT QUOTAS** button at the top of the page.
2. A panel will open on the right side.
3. Set the **New limit** to `1` (or however many you need) for both quotas.
4. Enter a **Request Description** (Justification). 
    * **Example Description**: *"Setting up an inference server to deploy the open-weights Flux.1 [dev] text-to-image model for development and testing. We require 1x NVIDIA L4 GPU in [Your Region] to handle the 24GB VRAM requirement of this 12B parameter model."*
5. Click **SUBMIT REQUEST**.

## 4. What Happens Next?
* **Automated Approval**: Sometimes, if your account has good billing history, small requests (like 1x L4) might be approved automatically within a few minutes.
* **Manual Review**: If it requires manual review, it may take 24-48 hours. You will receive an email update. Google Support might reply asking for payment in advance or more details about your company. If they do, clearly explain that you are testing generative AI models for development purposes.

## 5. Next Steps for Deployment
Once you receive the email confirming your quota is approved:
1. Go to **Compute Engine -> VM instances**.
2. Click **Create Instance**.
3. Select the region where you were approved.
4. Under **Machine configuration**, select **GPUs**.
5. Select your approved GPU. 
6. Scroll down to **Boot disk**, click **Change**, and switch the Operating System to **Deep Learning on Linux**. This will give you a pre-configured environment with CUDA drivers and PyTorch already installed, saving you hours of setup.
