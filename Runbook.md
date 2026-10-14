# MLOps assignment demo — command sheet

Repo: `https://github.com/2025ae05188/heart-disease-mlops`  
Project directory: `~/workspace/Assignment 1/heart-disease-mlops`

### Terminal 1 — install/check tools, get the project, prepare Docker + Minikube

```bash
cat /etc/os-release
```

```bash
if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y git curl ca-certificates docker conntrack-tools socat python3.12 python3.12-pip
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y git curl ca-certificates docker conntrack-tools socat python3.12 python3.12-pip
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git curl ca-certificates docker.io conntrack socat software-properties-common
  if ! apt-cache show python3.12-venv >/dev/null 2>&1; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
  fi
  sudo apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip
else
  echo "Unsupported package manager; stop here and install Git, curl, Docker, Python 3.12, conntrack and socat for this OS."
  exit 1
fi
```

```bash
python3.12 --version
sudo systemctl enable --now docker
```

If `docker info` reports **permission denied**, run `sudo usermod -aG docker "$USER"`, disconnect from the VM, reconnect, then continue.

```bash
docker info
```

```bash
if ! command -v minikube >/dev/null 2>&1; then
  curl -LO https://storage.googleapis.com/minikube/releases/v1.35.0/minikube-linux-amd64
  sudo install minikube-linux-amd64 /usr/local/bin/minikube
  rm minikube-linux-amd64
fi
minikube version
```

```bash
mkdir -p "$HOME/workspace/Assignment 1"
cd "$HOME/workspace/Assignment 1"
if [ -d heart-disease-mlops/.git ]; then
  cd heart-disease-mlops
  git switch main
  git pull --ff-only origin main
else
  git clone https://github.com/2025ae05188/heart-disease-mlops.git
  cd heart-disease-mlops
fi
pwd
git status --short
git log -1 --oneline
git remote -v
```

```bash
if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv
fi
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements-mlflow.txt -r requirements-dev.txt
python -m pip install -e .
python scripts/download_data.py
ruff check .
pytest
```

```bash
docker build -f src/Dockerfile -t heart-disease-api:1.1 .
minikube start --driver=docker
minikube status
minikube kubectl -- get nodes -o wide
minikube image load heart-disease-api:1.1
minikube addons enable ingress
minikube kubectl -- rollout status deployment/ingress-nginx-controller -n ingress-nginx --timeout=180s
```

Create the Grafana secret. Enter a password when prompted; the input is hidden. Remember it for the Grafana login.

```bash
read -r -s -p "Choose Grafana admin password: " GRAFANA_PASSWORD
printf '\n'
minikube kubectl -- create secret generic grafana-admin \
  --from-literal=admin-user=admin \
  --from-literal=admin-password="$GRAFANA_PASSWORD" \
  --dry-run=client -o yaml | minikube kubectl -- apply -f -
unset GRAFANA_PASSWORD
```

```bash
minikube kubectl -- apply -f k8s/deployment.yaml
minikube kubectl -- apply -f k8s/service.yaml
minikube kubectl -- apply -f k8s/ingress.yaml
minikube kubectl -- apply -f k8s/monitoring/prometheus.yaml
minikube kubectl -- apply -f k8s/monitoring/grafana-config.yaml
minikube kubectl -- apply -f k8s/monitoring/grafana.yaml
minikube kubectl -- rollout status deployment/heart-disease-api --timeout=180s
minikube kubectl -- rollout status deployment/prometheus --timeout=180s
minikube kubectl -- rollout status deployment/grafana --timeout=180s
minikube kubectl -- get pods -A -o wide
minikube kubectl -- get svc heart-disease-api prometheus grafana
minikube kubectl -- get ingress heart-disease-api
minikube kubectl -- get ingressclass
```

Check GitHub access:

```bash
git ls-remote origin main
git push --dry-run origin main
```

### Terminal 2 — start MLflow UI; leave it running

```bash
cd "$HOME/workspace/Assignment 1/heart-disease-mlops"
source .venv/bin/activate
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

### Terminal 3 — start Prometheus port-forward; leave it running

```bash
minikube kubectl -- port-forward svc/prometheus 9090:9090
```

### Terminal 4 — start Grafana port-forward; leave it running

```bash
minikube kubectl -- port-forward svc/grafana 3000:3000
```

Open these browser

- MLflow: `http://127.0.0.1:5000`
- GitHub Actions: `https://github.com/2025ae05188/heart-disease-mlops/actions`
- Prometheus targets: `http://127.0.0.1:9090/targets`
- Grafana dashboard: `http://127.0.0.1:3000/d/heart-disease-api/heart-disease-api-monitoring`

Use the VM's forwarded-port preview if the browser cannot reach the VM's `127.0.0.1` ports.

## 1. MLflow —

In Terminal 1:

```bash
cd "$HOME/workspace/Assignment 1/heart-disease-mlops"
source .venv/bin/activate
python scripts/run_experiment.py
```

Refresh the MLflow UI. Open experiment `heart-disease-classification`, then the newest run.

## 2. CI pipeline — trigger a new GitHub Actions run

In Terminal 1:

```bash
cd "$HOME/workspace/Assignment 1/heart-disease-mlops"
git status --short
git fetch origin
git status -sb
printf '%s\n' '# Serving package for the Heart Disease FastAPI application.' > src/heart_disease_ml/serving/__init__.py
git diff --check
git --no-pager diff -- src/heart_disease_ml/serving/__init__.py
git add src/heart_disease_ml/serving/__init__.py
git commit -m "chore: clarify serving package purpose"
git push origin main
git log -1 --oneline
```

Open the new Actions run for that commit. While it runs, continue with Section 3; return to Actions at the end to show the final result and uploaded `ci-training-artifacts` (`ci_model.pkl`, `metrics.json`).

## 3. API, Minikube, Prometheus and Grafana

In Terminal 1:

```bash
cd "$HOME/workspace/Assignment 1/heart-disease-mlops"
MINIKUBE_IP="$(minikube ip)"
echo "$MINIKUBE_IP"
minikube kubectl -- get pods -o wide
minikube kubectl -- get svc heart-disease-api prometheus grafana
minikube kubectl -- get ingress heart-disease-api
curl -i --max-time 10 -H 'Host: heart-disease' "http://${MINIKUBE_IP}/health"
```

```bash
curl -i --max-time 10 \
  -H 'Host: heart-disease' \
  -H 'Content-Type: application/json' \
  -X POST "http://${MINIKUBE_IP}/predict" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

```bash
for i in $(seq 1 10); do
  curl -sS --max-time 10 -o /dev/null \
    -w "prediction ${i}: HTTP %{http_code}\n" \
    -H 'Host: heart-disease' \
    -H 'Content-Type: application/json' \
    -X POST "http://${MINIKUBE_IP}/predict" \
    -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
done
curl -fsS -H 'Host: heart-disease' "http://${MINIKUBE_IP}/metrics" \
  | grep -E '^heart_disease_api_(requests_total|request_duration_seconds|requests_in_progress)' \
  | head -n 20
minikube kubectl -- logs deployment/heart-disease-api --tail=20
sleep 15
```

On the Prometheus page, enter these queries one at a time:

```promql
up{job="heart-disease-api"}
```

```promql
sum by (endpoint) (heart_disease_api_requests_total)
```

Open the Grafana dashboard and show all six panels. Return to the new GitHub Actions run and show its final status.
