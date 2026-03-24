# Azure Deployment Guide: Text-to-Image AI

Follow these step-by-step instructions to deploy your Text-to-Image application (with safety filter disabled) to a Microsoft Azure Virtual Machine.

## 1. Provision an Azure VM with GPU
To run Stable Diffusion efficiently, you need a GPU instance. The **NCas_T4_v3** series provides an NVIDIA T4 GPU which is very cost-effective.

1. Navigate to your Azure Portal -> **Virtual machines** -> **Create** -> **Azure virtual machine**.
2. **Virtual machine name**: `stable-diffusion-server`
3. **Region**: Choose a region with T4 availability (e.g., `East US`, `West US 2`, etc.).
4. **Image**: 
   - Search for **Data Science Virtual Machine - Ubuntu 20.04** or **Ubuntu 22.04 LTS**. (The Data Science VM comes with NVIDIA drivers and Docker pre-installed, highly recommended!)
5. **Size**: Select **Standard_NC4as_T4_v3** (4 vCPUs, 28 GiB memory, 1 NVIDIA T4 GPU).
   *(If you don't see it, check the "See all sizes" option or try a different region. Like AWS/GCP, you might need to request a quota increase for the NCv3/NCasT4 series, but it can be fast).*
6. **Administrator account**: Use **SSH public key** or **Password** as preferred.
7. **Inbound port rules**:
   - Public inbound ports: **Allow selected ports**
   - Select inbound ports: **SSH (22)**
8. Move to the **Disks** tab. Ensure your OS disk is at least **64 GiB**. The Docker image and Stable Diffusion weights take substantial space.
9. Move to the **Networking** tab. We will add a rule for port 7860 in the next step.
10. Click **Review + create**, then **Create**.

## 2. Open Port `7860` in Azure Network Security Group
1. Once deployed, click **Go to resource** (your new VM).
2. On the left menu, under **Settings**, click **Networking**.
3. Click **Add inbound port rule**.
4. **Source**: `Any`
5. **Source port ranges**: `*`
6. **Destination**: `Any`
7. **Destination port ranges**: `7860`
8. **Protocol**: `TCP`
9. **Action**: `Allow`
10. **Priority**: `100` (or any available number)
11. **Name**: `Port_7860`
12. Click **Add**.

## 3. Connect to the Instance
SSH into your Azure VM using the Public IP address shown on the Overview page:
```bash
ssh azureuser@<your-azure-public-ip>
```
*(Replace `azureuser` depending on what you named your admin account).*

If you chose the Data Science VM, Docker and NVIDIA drivers are pre-installed. You can verify by running `nvidia-smi` and `docker --version`. 

*(If you chose a basic Ubuntu image, you will need to install Docker and the NVIDIA Container Toolkit manually, using the same installation bash scripts provided in the AWS guide).*

## 4. Upload Your Code to the Server
Upload your files over SCP from your local machine (open a new local terminal):
```bash
scp requirements.txt app.py Dockerfile azureuser@<your-azure-public-ip>:/home/azureuser/
```

## 5. Build and Run the Docker Container
On your Azure VM, ensure you are in the directory containing your uploaded `Dockerfile`.

1. **Build the image**:
   *(Warning: This will take several minutes as it downloads PyTorch and the Stable Diffusion model weights.)*
   ```bash
   sudo docker build -t stable-diffusion-app .
   ```

2. **Run the container**:
   ```bash
   sudo docker run --gpus all -d -p 7860:7860 stable-diffusion-app
   ```

## 6. Access the Application
1. Find your VM's **Public IP address** in the Azure Portal.
2. Open any web browser and navigate to:
   ```
   http://<your-azure-public-ip>:7860
   ```
   *(Make sure you use `http://` and not `https://`)*

You are now running your text-to-image application on Azure!
