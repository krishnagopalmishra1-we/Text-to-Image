# AWS Deployment Guide: Text-to-Image AI

Follow these step-by-step instructions to deploy your Text-to-Image application (with filtered disabled) to an AWS EC2 instance.

## 1. Provision an EC2 Instance
To run Stable Diffusion efficiently, you need an instance with a GPU. The `g4dn` instance family is an excellent, cost-effective choice.

1. Navigate to your AWS Management Console -> **EC2** -> **Launch Instance**.
2. **Name**: `stable-diffusion-server`
3. **AMI (Amazon Machine Image)**: Search for and select an **AWS Deep Learning AMI GPU PyTorch** (based on Ubuntu). This AMI comes with NVIDIA drivers pre-installed, saving you the installation hassle. Alternatively, you can use a standard Ubuntu 22.04 LTS AMI and install drivers manually.
4. **Instance Type**: Select **`g4dn.xlarge`** (comes with 1 NVIDIA T4 GPU and 16GB RAM).
5. **Key Pair**: Create or select an existing key pair to allow SSH access.
6. **Network Settings**:
   - Ensure you allow SSH traffic from your IP.
   - **Crucial step:** Add a rule to **Allow Custom TCP on port `7860`** from anywhere (`0.0.0.0/0`). Our Gradio application runs on this port.
7. **Storage**: Allocate at least **60 GB** of root volume (`gp3`). The Docker image including PyTorch and the model weights will easily exceed 15-20 GB.
8. Click **Launch instance**.

## 2. Connect to the Instance and Prepare Environment
Once the instance is running, SSH into it from your local terminal:

```bash
ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>
```

If you chose an AMI that does **not** already have Docker and the NVIDIA Container Toolkit installed, you will need to run the following installation commands:

### Install Docker
```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```

### Install NVIDIA Container Toolkit
This allows Docker to communicate with the EC2 instance's GPU.
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 3. Upload Your Code to the Server
You can simply use `git` to clone a repository on the EC2 instance, or copy everything directly from your local machine using SCP over a new terminal window:

```bash
scp -i /path/to/your-key.pem requirements.txt app.py Dockerfile ubuntu@<your-ec2-public-ip>:/home/ubuntu/
```

## 4. Build and Run the Docker Container
On the EC2 instance, ensure you are in the same directory as the uploaded `Dockerfile`.

1. **Build the image**:
   *(Warning: This will take several minutes as it downloads the PyTorch base image and dependencies.)*
   ```bash
   sudo docker build -t stable-diffusion-app .
   ```

2. **Run the container**:
   ```bash
   sudo docker run --gpus all -d \
     --restart unless-stopped \
     --shm-size=2g \
     -v hf-cache:/tmp/hf-cache \
     -p 7860:7860 \
     stable-diffusion-app
   ```
   *Note the `--gpus all` flag—this gives the container direct access to the NVIDIA T4 GPU.*

## 5. Access the Application
The container will start quickly, and the first generation request will download model weights (one-time) into the cache volume. 

Open any web browser and navigate to:
```
http://<your-ec2-public-ip>:7860
```
*(Make sure you use `http://` and not `https://`)*

You should see your text-to-image application ready to use!
