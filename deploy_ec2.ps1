param(
    [Parameter(Mandatory = $true)]
    [string]$PublicIp,

    [Parameter(Mandatory = $true)]
    [string]$KeyPath,

    [string]$User = "ubuntu",
    [string]$RemoteDir = "/home/ubuntu/text-to-image",
    [string]$ImageName = "stable-diffusion-app",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $KeyPath)) {
    throw "Key file not found: $KeyPath"
}

$localFiles = @("app.py", "requirements.txt", "Dockerfile")
foreach ($file in $localFiles) {
    if (!(Test-Path $file)) {
        throw "Required file missing in current directory: $file"
    }
}

$sshTarget = "$User@$PublicIp"

Write-Host "Creating remote app directory..."
ssh -i $KeyPath -o StrictHostKeyChecking=accept-new $sshTarget "mkdir -p $RemoteDir"

Write-Host "Uploading files..."
scp -i $KeyPath -o StrictHostKeyChecking=accept-new app.py requirements.txt Dockerfile "$sshTarget`:$RemoteDir/"

$remoteLines = @(
    "set -e",
    "cd $RemoteDir",
    "if ! command -v docker >/dev/null 2>&1; then echo 'Docker not found. Install Docker first on this EC2 instance.'; exit 1; fi",
    "if ! docker info >/dev/null 2>&1; then echo 'Docker daemon not ready for current user. Retrying with sudo...'; fi",
    "sudo docker build -t $ImageName .",
    "sudo docker rm -f $ImageName >/dev/null 2>&1 || true",
    "sudo docker run --name $ImageName --gpus all -d --restart unless-stopped --shm-size=2g -v hf-cache:/tmp/hf-cache -p ${Port}:7860 $ImageName",
    "sleep 3",
    "sudo docker ps --filter name=$ImageName",
    "sudo docker logs --tail 120 $ImageName"
)
$remoteScript = ($remoteLines -join "`n")

Write-Host "Deploying container on EC2..."
ssh -i $KeyPath -o StrictHostKeyChecking=accept-new $sshTarget "bash -lc '$remoteScript'"

Write-Host "Deployment complete. Open: http://$PublicIp`:$Port"
