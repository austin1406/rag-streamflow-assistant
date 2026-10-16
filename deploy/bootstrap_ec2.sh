#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu 24.04 EC2 instance. Run it on the server as the default
# user (ubuntu):  bash bootstrap_ec2.sh
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y ca-certificates curl

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

# 2 GB swap so the embedding model has headroom on a small instance.
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

sudo mkdir -p /opt/rag
sudo chown "$USER":"$USER" /opt/rag
touch /opt/rag/.env
chmod 600 /opt/rag/.env

echo "Done. Add your LLM key to /opt/rag/.env (see DEPLOY.md), then log out and back in."
