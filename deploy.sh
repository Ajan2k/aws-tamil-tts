#!/bin/bash
# One-click deploy script for EC2 Ubuntu

set -e

echo "=== Tamil Chennai TTS AWS Deploy ==="
echo "Region: ap-south-1 recommended for Chennai latency"

# Check docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt update
    sudo apt install -y docker.io docker-compose-plugin ffmpeg espeak-ng
    sudo usermod -aG docker $USER
    echo "Docker installed. Please logout/login and run again if needed."
fi

# Check nvidia for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi
    echo "Installing nvidia-container-toolkit if not present..."
    # For g4dn/g5 instances
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-container-toolkit/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-container-toolkit/$distribution/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit || echo "nvidia toolkit install failed, continuing"
    sudo systemctl restart docker || true
else
    echo "No GPU detected - will run on CPU (ok for t3.medium test)"
fi

echo "Building and starting..."
docker compose up --build -d

echo "Waiting 20s for model load..."
sleep 20
docker logs tamil-tts --tail 50

PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "localhost")
echo ""
echo "=== Deployed ==="
echo "API: http://$PUBLIC_IP:8000"
echo "Docs: http://$PUBLIC_IP:8000/docs"
echo "Health: curl http://$PUBLIC_IP:8000/health"
echo ""
echo "Test:"
echo "curl -X POST http://$PUBLIC_IP:8000/synthesize -H 'Content-Type: application/json' -d '{\"text\":\"Vanakkam da machan\"}' --output test.wav"
