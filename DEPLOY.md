# Deploying to AWS EC2

Every push to `master` that passes the **Tests** workflow triggers **Deploy**
(`.github/workflows/deploy.yml`):

1. **build**: builds the Docker image (vector index and embedding model baked in) and pushes
   it to GitHub Container Registry as `ghcr.io/austin1406/rag-streamflow-assistant`, tagged
   `latest` and with the commit SHA.
2. **deploy**: SSHes into your EC2 instance, pulls that exact SHA, restarts the container, and
   waits for `/health` to pass. If `EC2_HOST` / `EC2_SSH_KEY` are not set, this step is skipped
   and the build still succeeds.

## One-time setup

### 1. Launch the server (AWS console)

- Create an AWS account and set a **billing alert** first (Billing > Budgets).
- Launch an EC2 instance: **Ubuntu 24.04**, instance type **t3.small** (2 GB RAM). A 1 GB
  instance such as t3.micro is too small for the embedding model. Check AWS's current
  free-tier / credit terms before choosing.
- Create a key pair and download the `.pem` file.
- Security group: allow **SSH (22)** from your IP and **HTTP (80)** from anywhere.

### 2. Prepare the server

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
curl -fsSLO https://raw.githubusercontent.com/austin1406/rag-streamflow-assistant/master/deploy/bootstrap_ec2.sh
bash bootstrap_ec2.sh
```

Then pick an LLM backend. The server cannot run Ollama on a small instance, so use a hosted
provider (Groq has a free tier). Edit `/opt/rag/.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=<your key>
```

Log out and back in so the `docker` group applies.

### 3. Add GitHub secrets

Repo > Settings > Secrets and variables > Actions:

| Secret | Value |
|---|---|
| `EC2_HOST` | the instance's public IP or DNS name |
| `EC2_SSH_KEY` | the full contents of the `.pem` file |
| `EC2_USER` | optional, defaults to `ubuntu` |

### 4. Deploy

Push to `master` (or run **Deploy** manually from the Actions tab). Open
`http://<EC2_PUBLIC_IP>/` and `http://<EC2_PUBLIC_IP>/health`.

## Notes

- The first SSH connection trusts the server's host key automatically
  (`StrictHostKeyChecking=accept-new`); for stricter setups, pin the key in a secret.
- API keys live only in `/opt/rag/.env` on the server, never in the image or the repo
  (`.dockerignore` excludes `.env`).
- To stop paying, stop or terminate the instance in the AWS console.
- To test the image on your own machine: `docker build -t rag . && docker run --rm -p 8000:8000 --env-file .env rag`
  (the image is about 2 GB).

## Stopping and restarting the server

Stopping the instance in the AWS console (Instance state > Stop) ends compute charges; only the
20 GB disk keeps costing a little. Everything on the disk survives, including Docker, the app
image, and `/opt/rag/.env`.

**To start it again:**

1. EC2 > Instances > select the instance > **Instance state > Start instance**.
2. Wait for **Running** and copy the new **Public IPv4 address** (it changes on every start unless
   you attach an Elastic IP).
3. Update the GitHub secret so future deploys find the server:
   `gh secret set EC2_HOST --body "<new ip>"`
4. The container restarts by itself (`--restart unless-stopped`). After a minute or two, open
   `http://<new ip>/health`. If it does not respond, SSH in and run `docker ps`, then
   `docker start rag`, or re-run the **Deploy** workflow from the Actions tab.

While the server is stopped, pushes to `master` still build and publish the image; the deploy
step notices the server is unreachable and skips instead of failing.

**If you terminated the instance instead of stopping it,** start over from "One-time setup"
above: launch a new instance, run `deploy/bootstrap_ec2.sh`, create `/opt/rag/.env`, update the
`EC2_HOST` and `EC2_SSH_KEY` secrets, and run the Deploy workflow.
