## Deployment Options:


# **🏢 The AI Skyscraper: Kubernetes Production Deployment!**

Good morning class! Today we're looking at the **NASA-level** deployment configuration for our Financial AI - a Kubernetes manifest! This isn't just running our AI; this is building an **entire AI data center** in code!

---

## **🚀 What This YAML Does**

This single file creates a **complete production environment** that:
1. **🏗️ Builds a secure neighborhood** (Namespace)
2. **📋 Sets all configurations** (ConfigMaps & Secrets)
3. **🏭 Deploys our AI factory** (Deployment with 3 copies)
4. **🚪 Creates access doors** (Services)
5. **📊 Adds auto-scaling** (Horizontal Pod Autoscaler)
6. **🔐 Secures with HTTPS** (Ingress with TLS)
7. **💾 Provides storage** (Persistent Volumes)

**Think of it like:** Blueprints for an entire AI city that builds itself!

---

## Step 16: Kubernetes Deployment Manifests









### Create `kubernetes/namespace.yaml`
<!-- 
### **1. The Neighborhood (Namespace)**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: financial-rag
```
- Creates a **separate area** in Kubernetes
- Like a "gated community" for our app
- **Isolation:** Other apps can't interfere
- **Organization:** All our stuff lives together -->

---

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: financial-rag
  labels:
    name: financial-rag
    environment: production
```

### Create `kubernetes/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: financial-rag-config
  namespace: financial-rag
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  EMBEDDING_MODEL: "all-MiniLM-L6-v2"
  LLM_MODEL: "gpt-3.5-turbo"
  CHUNK_SIZE: "1000"
  CHUNK_OVERLAP: "200"
  TOP_K_RESULTS: "3"
  VECTOR_STORE_PATH: "/app/data/chroma_db"
  RAW_DATA_PATH: "/app/data/raw"
  PROCESSED_DATA_PATH: "/app/data/processed"

#   - Stores **non-secret configuration**
# - Can be changed without rebuilding containers
# - **Environment-specific:** Dev vs Prod settings
# - **Shared by all pods**

# **Why `all-MiniLM-L6-v2` for production?**
# - Local embedding model (no API costs!)
# - Good enough for many use cases
# - No external dependencies

```

### Create `kubernetes/secret.yaml`

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: financial-rag-secrets
  namespace: financial-rag
type: Opaque
stringData:
  OPENAI_API_KEY: ""  # Will be filled from CI/CD
  WANDB_API_KEY: ""   # Will be filled from CI/CD


# - Stores **secrets** (API keys, passwords)
# - Base64 encoded (not really secure, but better than plaintext)
# - **Never in code or ConfigMaps!**
# - **CI/CD fills these** during deployment

```

### Create `kubernetes/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-rag-api
  namespace: financial-rag
  labels:
    app: financial-rag-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: financial-rag-api
# - **3 identical copies** of our app
# - **High availability:** If one fails, two others serve traffic
# - **Load distribution:** Share the workload

  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
# - **Zero-downtime updates!**
# - New version rolls out gradually
# - Old version stays up until new is ready

  template:
    metadata:
      labels:
        app: financial-rag-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: financial-rag-api
        image: financial-rag-agent:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: financial-rag-secrets
              key: OPENAI_API_KEY
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: financial-rag-secrets
              key: WANDB_API_KEY
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: LOG_LEVEL
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: ENVIRONMENT
        - name: EMBEDDING_MODEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: EMBEDDING_MODEL
        - name: LLM_MODEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: LLM_MODEL

# **Requests = Guaranteed resources**
# - Kubernetes reserves this much
# - **Like:** "Reserving a conference room"

# **Limits = Maximum allowed**
# - Container killed if exceeds
# - **Prevents:** "Noisy neighbor" problems
        resources:
            requests:
                memory: "1Gi"   # Guaranteed 1GB RAM
                cpu: "500m"     # Guaranteed 0.5 CPU
            limits:
                memory: "2Gi"   # Never more than 2GB
                cpu: "1000m"    # Never more than 1 CPU

# **Liveness Probe:** "Is the app alive?"
# - If `/health` fails 3× → Container gets restarted
# - **Self-healing system!**
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

# **Readiness Probe:** "Is the app ready for traffic?"
# - If not ready → Remove from load balancer
# - **Prevents sending requests to broken pods**

        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
        volumeMounts:
        - name: data-storage
          mountPath: /app/data
        - name: log-storage
          mountPath: /app/logs
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: financial-rag-pvc
      - name: log-storage
        emptyDir: {}
      restartPolicy: Always
```

### Create `kubernetes/service.yaml`
<!-- - **Internal access only** (within Kubernetes)
- Other services can talk to our AI
- **Like:** Employee-only entrance -->
```yaml
apiVersion: v1
kind: Service
metadata:
  name: financial-rag-service
  namespace: financial-rag
  labels:
    app: financial-rag-api
spec:
  selector:
    app: financial-rag-api
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: financial-rag-service-external
  namespace: financial-rag
  labels:
    app: financial-rag-api
spec:
  selector:
    app: financial-rag-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: LoadBalancer

#   - **Public internet access**
# - Cloud creates a load balancer
# - **Like:** Main customer entrance
```

### Create `kubernetes/persistent-volume-claim.yaml`

```yaml

# - **10GB of persistent storage**
# - Vector database survives pod restarts
# - **Not in container** (containers are ephemeral!)
# - **Mounts to:** `/app/data` in our containers

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: financial-rag-pvc
  namespace: financial-rag
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # Adjust based on your Kubernetes cluster
```

### Create `kubernetes/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: financial-rag-hpa
  namespace: financial-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: financial-rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
# **Magic auto-scaling:**
# - Normally: 2 pods running
# - If CPU > 70% for 5 minutes → Add pods
# - Up to 10 pods max!
# - When load drops → Remove pods

  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100 #can double every minute
        periodSeconds: 60

# - **Scale up fast** (traffic spike!)
# - **Scale down slow** (avoid thrashing)
```

### Create `kubernetes/ingress.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: financial-rag-ingress
  namespace: financial-rag
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - financial-rag.yourcompany.com
    secretName: financial-rag-tls
  rules:
  - host: financial-rag.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: financial-rag-service
            port:
              number: 8000

# **Production-grade setup:**
# - **Custom domain:** `financial-rag.yourcompany.com`
# - **HTTPS automatically** (Let's Encrypt via cert-manager)
# - **SSL redirect:** HTTP → HTTPS
# - **Path routing:** Can add more paths later
```





---

## **🎓 Real-World Scenarios**

### **Scenario 1: Black Friday Sale (Traffic Spike)**
```
9 AM: 2 pods, CPU 40%
10 AM: Traffic spikes! CPU 85% 
10:01 AM: HPA detects high CPU
10:02 AM: Scales to 4 pods, CPU 60%
10:05 AM: Scales to 6 pods, CPU 50%
Peak handled! No downtime!
```

### **Scenario 2: Pod Crash**
```
Pod #2 crashes (bug in code)
Liveness probe fails 3×
Kubernetes kills pod, starts new one
Load balancer sends traffic to pods #1 and #3
Users don't notice!
```

### **Scenario 3: Configuration Update**
```
Change LOG_LEVEL from INFO to DEBUG
Update ConfigMap
Kubernetes automatically updates all pods
No restart needed (env variables update)
```

---

## **🔍 The Magic of Kubernetes**

### **Self-Healing:**
```yaml
# If health checks fail → Restart
# If node dies → Reschedule on another node
# If disk full → Clean up or expand
```

### **Declarative Configuration:**
```yaml
# You DECLARE what you want:
replicas: 3
# Kubernetes MAKES it happen
# Always tries to match declared state
```

### **Service Discovery:**
```yaml
# Pods get dynamic IPs
# Service provides stable DNS name:
financial-rag-service.financial-rag.svc.cluster.local
# Other apps just use the name!
```

---

## **💡 Classroom Activities**

### **Activity 1: The Scaling Game**
```bash
# Simulate traffic with hey/ab
hey -n 1000 -c 50 https://financial-rag.yourcompany.com/analyze

# Watch pods scale:
kubectl get hpa financial-rag-hpa -w
# NAME              REFERENCE                TARGETS   MINPODS   MAXPODS
# financial-rag-hpa Deployment/financial...  45%/70%   2         10
```

### **Activity 2: The Failure Injection**
```python
# Add random failures to health endpoint
import random

@app.get("/health")
def health():
    if random.random() < 0.3:
        return {"status": "unhealthy"}, 500
    # Watch Kubernetes restart pods!
```

### **Activity 3: The Configuration Rollout**
```bash
# Change ConfigMap
kubectl edit configmap financial-rag-config
# Change LOG_LEVEL to "DEBUG"

# Watch logs change
kubectl logs -f deployment/financial-rag-api
# [INFO] → [DEBUG] without restart!
```

---

## **⚡ Production Pro Tips**

### **1. Resource Optimization:**
```yaml
# Start small, monitor, adjust
resources:
  requests:
    memory: "512Mi"  # Might be enough
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### **2. Pod Disruption Budget:**
```yaml
# Add to prevent too many pods down at once
apiVersion: policy/v1
kind: PodDisruptionBudget
spec:
  minAvailable: 1  # Always at least 1 pod up
  selector:
    matchLabels:
      app: financial-rag-api
```

### **3. Network Policies:**
```yaml
# Restrict which pods can talk to our AI
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
spec:
  podSelector:
    matchLabels:
      app: financial-rag-api
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend-namespace
  # Only frontend can access!
```

---

## **🎯 Key Takeaways**

1. **Kubernetes = Data center in code** - Self-healing, auto-scaling
2. **Multiple replicas = High availability** - No single point of failure
3. **Probes = Self-healing** - Automatically fix problems
4. **HPA = Cost optimization** - Pay for what you use
5. **ConfigMaps/Secrets = Security** - Separate config from code

**This transforms our AI from:**
- **"A program on a server"** → **"A resilient cloud service"**
- **"Manual scaling"** → **"Automatic elasticity"**
- **"Scheduled maintenance"** → **"Zero-downtime updates"**

**Question for discussion:** If you were to add one more Kubernetes feature to make this even more production-ready, what would it be and why?








We could deploy this a variety of ways.

Option A: Local Development
```bash
python scripts/start_api.py
```

Option B: Docker
```bash
docker-compose up -d
```

Option C: Kubernetes (Next Step)
We can create Kubernetes manifests for full enterprise deployment.

**Run the production test and let me know the results!** This will validate that our entire system is enterprise-ready.

The system is now capable of:
- Scalable REST API with proper documentation
- Containerized deployment
- Comprehensive monitoring and observability
- Health checks and metrics
- Background processing
- Production-grade error handling

Ready for the final step: Kubernetes deployment manifests?

Perfect! Let's complete the enterprise deployment with Kubernetes manifests and final production optimizations.

Kubernetes Deployment Manifests

Create `kubernetes/namespace.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: financial-rag
  labels:
    name: financial-rag
    environment: production
```

Create `kubernetes/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: financial-rag-config
  namespace: financial-rag
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  EMBEDDING_MODEL: "all-MiniLM-L6-v2"
  LLM_MODEL: "gpt-3.5-turbo"
  CHUNK_SIZE: "1000"
  CHUNK_OVERLAP: "200"
  TOP_K_RESULTS: "3"
  VECTOR_STORE_PATH: "/app/data/chroma_db"
  RAW_DATA_PATH: "/app/data/raw"
  PROCESSED_DATA_PATH: "/app/data/processed"
```

Create `kubernetes/secret.yaml`

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: financial-rag-secrets
  namespace: financial-rag
type: Opaque
stringData:
  OPENAI_API_KEY: ""  # Will be filled from CI/CD
  WANDB_API_KEY: ""   # Will be filled from CI/CD
```

Create `kubernetes/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-rag-api
  namespace: financial-rag
  labels:
    app: financial-rag-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: financial-rag-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: financial-rag-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: financial-rag-api
        image: financial-rag-agent:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: financial-rag-secrets
              key: OPENAI_API_KEY
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: financial-rag-secrets
              key: WANDB_API_KEY
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: LOG_LEVEL
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: ENVIRONMENT
        - name: EMBEDDING_MODEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: EMBEDDING_MODEL
        - name: LLM_MODEL
          valueFrom:
            configMapKeyRef:
              name: financial-rag-config
              key: LLM_MODEL
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
        volumeMounts:
        - name: data-storage
          mountPath: /app/data
        - name: log-storage
          mountPath: /app/logs
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: financial-rag-pvc
      - name: log-storage
        emptyDir: {}
      restartPolicy: Always
```

Create `kubernetes/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: financial-rag-service
  namespace: financial-rag
  labels:
    app: financial-rag-api
spec:
  selector:
    app: financial-rag-api
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: financial-rag-service-external
  namespace: financial-rag
  labels:
    app: financial-rag-api
spec:
  selector:
    app: financial-rag-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: LoadBalancer
```

Create `kubernetes/persistent-volume-claim.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: financial-rag-pvc
  namespace: financial-rag
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # Adjust based on your Kubernetes cluster
```

Create `kubernetes/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: financial-rag-hpa
  namespace: financial-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: financial-rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

Create `kubernetes/ingress.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: financial-rag-ingress
  namespace: financial-rag
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - financial-rag.yourcompany.com
    secretName: financial-rag-tls
  rules:
  - host: financial-rag.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: financial-rag-service
            port:
              number: 8000
```

Now let's add continuous integration anmd continuous deployment
CI/CD Pipeline Configuration

Create `.github/workflows/ci-cd.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  IMAGE_NAME: financial-rag-agent
  REGISTRY: ghcr.io

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests
      run: |
        python test_foundation.py
        python test_agent.py
        python test_production.py
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    - name: Run security scan
      run: |
        pip install bandit safety
        bandit -r src/ -f json -o bandit-report.json
        safety check --json

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Log in to GitHub Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata (tags, labels)
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ github.repository }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_STAGING }}
        command: apply -f kubernetes/
        version: v1.27.0

    - name: Verify deployment
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_STAGING }}
        command: rollout status deployment/financial-rag-api -n financial-rag
        version: v1.27.0

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_PRODUCTION }}
        command: apply -f kubernetes/
        version: v1.27.0

    - name: Verify deployment
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_PRODUCTION }}
        command: rollout status deployment/financial-rag-api -n financial-rag
        version: v1.27.0
```



## Step 17: CI/CD Pipeline Configuration


# **🏭 The AI Assembly Line: Automated CI/CD Pipeline!**

Good morning class! Today we're looking at the **most advanced and professional** part of software engineering - the CI/CD pipeline! This is like having a **fully automated AI factory** that builds, tests, and deploys our Financial Agent every time we make a change!

---

## **🚀 What This Pipeline Does**

This GitHub Actions workflow creates an **automated assembly line** that:
1. **🧪 Tests everything** when code changes
2. **🏗️ Builds a new Docker image** if tests pass
3. **🚀 Deploys to staging** for validation
4. **🌍 Deploys to production** after staging success
5. **🔄 Runs automatically** on every code change

**Think of it like:** A self-driving car factory that builds, quality-checks, and ships cars automatically!

---

## **🔧 The 4-Stage Assembly Line**

### **Stage 1: The Quality Control (Test)**
### **Stage 2: The Factory Floor (Build)**
### **Stage 3: The Showroom (Staging)**
### **Stage 4: The Customer Delivery (Production)**

Each stage **must pass** before moving to the next!

---







### **Step 1: Checkout Code**
```yaml
- uses: actions/checkout@v4
```
<!-- - **Git clone** our repository
- Uses GitHub's optimized checkout action
- **Version @v4** = Latest stable -->

<!-- ### **Step 2: Python Setup** -->
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: ${{ matrix.python-version }}
```
<!-- - Installs **Python 3.9**
- Uses GitHub's pre-built Python
- **Fast:** No compiling Python from source -->

<!-- ### **Step 3: Install Dependencies** -->
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -e .[dev]
```
<!-- **Two commands:**
1. **Upgrade pip** (always latest)
2. **Install our package** with `[dev]` extras
   - `-e .` = "Editable" install (links to code)
   - `[dev]` = Development dependencies (pytest, etc.) -->

<!-- ### **Step 4: Run Tests** -->
```yaml
- name: Run tests
  run: |
    python test_foundation.py
    python test_agent.py
    python test_production.py
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```
<!-- **Three test suites:**
1. `test_foundation.py` = Core components work
2. `test_agent.py` = AI agent works
3. `test_production.py` = Production readiness -->

<!-- **Secret environment variable:** -->
- `${{ secrets.OPENAI_API_KEY }}` = From GitHub Secrets
<!-- - **Never in code!** Secure storage
- Tests need real API key -->

<!-- ### **Step 5: Security Scan** -->
```yaml
- name: Run security scan
  run: |
    pip install bandit safety
    bandit -r src/ -f json -o bandit-report.json
    safety check --json
```
<!-- **Two security tools:**
1. **Bandit** = Python code security scanner
   - Finds hardcoded passwords, SQL injection risks
2. **Safety** = Dependency vulnerability scanner
   - Checks if any packages have known security issues -->

<!-- **Professional practice:** Security scanning in CI! -->

---

## **🏗️ Job 2: The Factory Floor (Build & Push)**

### **Dependency:**
```yaml
needs: test
if: github.ref == 'refs/heads/main'
```
<!-- - **`needs: test`** = Only runs if tests pass!
- **`if: main`** = Only on main branch (not PRs or develop) -->

### **Step 1: Docker Login**
```yaml
- name: Log in to GitHub Container Registry
  uses: docker/login-action@v2
  with:
    registry: ${{ env.REGISTRY }}
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}
```
<!-- **Authenticates to GitHub Container Registry:**
- `${{ github.actor }}` = Who triggered the workflow
- `${{ secrets.GITHUB_TOKEN }}` = Automatic GitHub token
- **Secure:** No manual passwords needed -->

### **Step 2: Smart Tagging**
```yaml
- name: Extract metadata (tags, labels)
  id: meta
  uses: docker/metadata-action@v4
  with:
    images: ghcr.io/username/repo/financial-rag-agent
    tags: |
      type=ref,event=branch    # main → latest
      type=ref,event=pr        # pr-123 → pr-123
      type=semver,pattern={{version}}  # v1.2.3 → 1.2.3
      type=sha,prefix={{branch}}-  # main-abc123
```
<!-- **Automatic tagging magic:**
- **Branch-based:** `main` → `latest`
- **PR-based:** `pr-123` → `pr-123`
- **Semantic version:** `v1.2.3` → `1.2.3`, `1.2`, `1`
- **Commit-based:** `main-abc123def` -->

**Example tags created:**
```
<!-- ghcr.io/yourname/financial-rag-agent:latest
ghcr.io/yourname/financial-rag-agent:main
ghcr.io/yourname/financial-rag-agent:1.2.3
ghcr.io/yourname/financial-rag-agent:1.2
ghcr.io/yourname/financial-rag-agent:1
ghcr.io/yourname/financial-rag-agent:main-abc123def -->
```

### **Step 3: Build with Cache**
```yaml
- name: Build and push Docker image
  uses: docker/build-push-action@v4
  with:
    context: .
    push: true
    tags: ${{ steps.meta.outputs.tags }}
    labels: ${{ steps.meta.outputs.labels }}
    cache-from: type=gha
    cache-to: type=gha,mode=max
<!-- ```
**Smart building:**
- `cache-from: type=gha` = Use GitHub Actions cache
- **Fast builds:** Layers cached between runs
- `push: true` = Push to registry automatically
- `labels` = Metadata about build (git sha, date, etc.) -->

---

## **🚀 Job 3: The Showroom (Staging Deployment)**

### **Dependencies:**
```yaml
needs: build-and-push
environment: staging
```
<!-- - **`needs: build-and-push`** = After successful build
- **`environment: staging`** = GitHub environment protection
  - Can require approvals
  - Separate secrets
  - Audit log -->

### **Step 1: Deploy to Kubernetes**
```yaml
- name: Deploy to Kubernetes
  uses: steebchen/kubectl@v2
  with:
    config: ${{ secrets.KUBECONFIG_STAGING }}
    command: apply -f kubernetes/
```
<!-- **What happens:**
1. Uses `kubectl` GitHub Action
2. Authenticates with `KUBECONFIG_STAGING` secret
3. Runs `kubectl apply -f kubernetes/`
4. **Applies ALL Kubernetes manifests** from `kubernetes/` folder -->

### **Step 2: Verify Deployment**
```yaml
- name: Verify deployment
  uses: steebchen/kubectl@v2
  with:
    config: ${{ secrets.KUBECONFIG_STAGING }}
    command: rollout status deployment/financial-rag-api
```
<!-- **Waits for:**
- Pods to start
- Health checks to pass
- Service to be ready
- **If fails:** Pipeline fails (prevents bad deployments!) -->

---

## **🌍 Job 4: Customer Delivery (Production)**

### **The Gatekeeper:**
```yaml
needs: deploy-staging
environment: production
if: github.ref == 'refs/heads/main'
```
<!-- **Three safety checks:**
1. **`needs: deploy-staging`** = Staging must pass first
2. **`environment: production`** = Extra protection layer
3. **`if: main`** = Only from main branch -->

<!-- **Production environment typically requires:**
- Manual approval
- Multiple reviewers
- Business hours only -->

<!-- ### **Same steps as staging:**
- Deploy to production Kubernetes
- Verify deployment
- **But with `KUBECONFIG_PRODUCTION` secret!** -->

---

### **Scenario 1: Developer Makes a Change**

1. Developer pushes to "develop" branch
2. Pipeline runs tests ONLY (no deploy)
3. Tests pass → Developer creates PR to main
4. Pipeline runs tests on PR
5. Team reviews, approves, merges
6. Pipeline: Test → Build → Deploy staging → Deploy production
7. New feature live in 10 minutes!


### **Scenario 2: Security Vulnerability Found**

1. Safety scan finds vulnerable dependency
2. Pipeline FAILS at test job
3. Developer gets notification
4. Fix dependency, push again
5. Pipeline passes, safe deployment continues


### **Scenario 3: Rollback Needed**

1. Production deployment has bug
2. Developer: git revert last commit
3. Push to main
4. Pipeline builds OLD working version
5. Deploys it automatically
6. Bug fixed in 5 minutes!


---

## **🔍 The Magic of GitHub Actions**

### **Secrets Management:**
yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```
- **Never exposed** in logs
- **Encrypted at rest**
- **Different per environment** (staging vs production)

### **Matrix Builds (Future Enhancement):**
```yaml
strategy:
  matrix:
    python-version: [3.9, 3.10, 3.11]
    os: [ubuntu-latest, macos-latest]
```
**Would test:**
- Python 3.9 on Ubuntu
- Python 3.9 on MacOS
- Python 3.10 on Ubuntu
- etc.
**Ensures cross-platform compatibility!**

### **Artifacts (Future Enhancement):**
```yaml
- name: Upload test results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-results.xml
```
**Store:** Test reports, security scans, coverage data

---

## **💡 Classroom Activities**

### **Activity 1: The Pipeline Detective**
```bash
# Given a failed pipeline, students investigate:
1. Which job failed?
2. Which step failed?  
3. What do the logs say?
4. How would they fix it?

# Example: Test failure → Check test logs
# Example: Build failure → Check Dockerfile
# Example: Deploy failure → Check kubeconfig
```

### **Activity 2: The Secret Keeper**
```yaml
# Task: "Add a new secret"
# Students would:
1. Add to GitHub Secrets: DATABASE_PASSWORD
2. Update pipeline to use it
3. Update app to read it
4. Test deployment

# Learn: Secure secret management
```

### **Activity 3: The Rollback Drill**
```bash
# Simulate a bad deployment
1. Add bug: @app.get("/health") return 500
2. Commit, push to main
3. Watch pipeline deploy bug
4. Health checks fail in production!
5. Rollback: git revert, push
6. Watch pipeline restore working version
```


## **🎯 Key Takeaways**

1. **CI/CD = Automated quality** - No manual deployments!
2. **Stages = Safety gates** - Test → Build → Stage → Prod
3. **Secrets = Secure** - Never in code, encrypted storage
4. **GitHub Actions = Powerful** - Free for open source!
5. **Rolling updates = Zero downtime** - Users never see deployment

**This transforms development from:**
- **"Manual, error-prone"** → **"Automated, reliable"**
- **"Fear of deployment"** → **"Confident, frequent updates"**
- **"Long release cycles"** → **"Minutes from code to production"**

**Question for discussion:** If you could add one more check or step to this pipeline to make it even more robust, what would it be and why?

### Create `.github/workflows/ci-cd.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]  # Code push to main or develop
  pull_request:
    branches: [ main ] # PR to main branch

#     **Two triggers:**
# 1. **Push to main/develop** → "Deploy new features"
# 2. **Pull request to main** → "Test before merging"

# **Why both?**
# - **PRs:** Catch bugs BEFORE they reach main
# - **Push to develop:** Test integration
# - **Push to main:** Deploy to production

env:
  IMAGE_NAME: financial-rag-agent
  REGISTRY: ghcr.io  # GitHub Container Registry

  
# **Shared across all jobs:**
# - `IMAGE_NAME` = Our Docker image name
# - `REGISTRY` = Where to store images (GitHub)
# - **Consistency:** Same values everywhere

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9] # Could add [3.9, 3.10, 3.11]

# - **Ubuntu Linux** runner
# - **Matrix testing:** Test on multiple Python versions
# - **Future:** Add 3.10, 3.11 for compatibility

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests
      run: |
        python test_foundation.py
        python test_agent.py
        python test_production.py
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    - name: Run security scan
      run: |
        pip install bandit safety
        bandit -r src/ -f json -o bandit-report.json
        safety check --json

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Log in to GitHub Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata (tags, labels)
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ github.repository }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_STAGING }}
        command: apply -f kubernetes/
        version: v1.27.0

    - name: Verify deployment
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_STAGING }}
        command: rollout status deployment/financial-rag-api -n financial-rag
        version: v1.27.0

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_PRODUCTION }}
        command: apply -f kubernetes/
        version: v1.27.0

    - name: Verify deployment
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBECONFIG_PRODUCTION }}
        command: rollout status deployment/financial-rag-api -n financial-rag
        version: v1.27.0
```














## Step 18: Monitoring and Metrics


# **📊 The AI Scoreboard: Real-Time Metrics and Monitoring!**

Good morning class! Today we're looking at the **dashboard and scoreboard** for our Financial AI - the metrics collection system! This is how we track every single thing our AI does, in real-time, to understand performance, costs, and reliability!

---

## **🚀 What This Code Does**

This code creates a **comprehensive monitoring system** that:
1. **📈 Tracks every query** (success/failure, duration)
2. **🔧 Monitors tool usage** (which tools work/fail)
3. **🧮 Counts AI "brain power"** (token usage = costs!)
4. **📚 Measures knowledge base** (vector store size)
5. **📊 Exposes metrics** for Prometheus (industry standard)

**Think of it like:** The control panel of a spaceship, showing speed, fuel, systems status, and everything happening inside!

---

## **🎯 Why Metrics Matter for AI Systems**

### **Without Metrics:**
- "Is our AI working?" → "I think so?"
- "How much does it cost?" → "No idea!"
- "Which tools fail most?" → "Not sure!"
- "When should we scale?" → "When users complain!"

### **With Metrics:**
- **Success rate:** 97.3% (great!)
- **Average cost:** $0.012 per query
- **Slowest tool:** `get_stock_price` (2.1s avg)
- **Vector store:** 1,542 documents
- **Alert:** Success rate dropped to 85%!

---

## **🔧 The 5 Types of Metrics**

### **1. The Query Counter (Success/Failure Tracking)**
```python
QUERY_COUNTER = Counter('financial_rag_queries_total', 
                       'Total number of queries', 
                       ['status', 'agent_type'])
```
**What it tracks:**
- **Total queries** (increments every query)
- **Labels:** `status` = "success" or "failure"
- **Labels:** `agent_type` = "tool_using_agent" or "simple_rag"

**Example data:**
```
financial_rag_queries_total{status="success", agent_type="tool_using_agent"} 142
financial_rag_queries_total{status="failure", agent_type="tool_using_agent"} 8
financial_rag_queries_total{status="success", agent_type="simple_rag"} 45
```

**You can calculate:**
- **Success rate:** 142 / (142+8) = 94.7%
- **Tool agent vs simple usage:** 150 vs 45 queries

---

### **2. The Stopwatch (Query Duration)**
```python
QUERY_DURATION = Histogram('financial_rag_query_duration_seconds',
                          'Query duration in seconds')
```
**Histogram = Bucketed timing:**
- Groups queries by how long they took
- **Example buckets:** 0.1s, 0.5s, 1s, 5s, 10s, 30s

**Example data:**
```
financial_rag_query_duration_seconds_bucket{le="0.1"} 15
financial_rag_query_duration_seconds_bucket{le="0.5"} 42
financial_rag_query_duration_seconds_bucket{le="1.0"} 89
financial_rag_query_duration_seconds_bucket{le="5.0"} 145
financial_rag_query_duration_seconds_bucket{le="+Inf"} 150
```

**You can calculate:**
- **95th percentile:** 3.2s (95% of queries under 3.2s)
- **Median:** 0.8s
- **Slow queries:** 5 queries took >5 seconds

---

### **3. The Tool Tracker (Which Tools Work)**
```python
AGENT_TOOL_USAGE = Counter('financial_rag_agent_tool_usage_total',
                          'Agent tool usage',
                          ['tool_name', 'status'])
```
**Tracks each tool:**
- `tool_name` = "get_stock_price", "search_filings", etc.
- `status` = "success" or "failure"

**Example data:**
```
financial_rag_agent_tool_usage_total{tool_name="get_stock_price", status="success"} 87
financial_rag_agent_tool_usage_total{tool_name="get_stock_price", status="failure"} 12  # High failure!
financial_rag_agent_tool_usage_total{tool_name="search_filings", status="success"} 145
financial_rag_agent_tool_usage_total{tool_name="search_filings", status="failure"} 2
```

**Actionable insight:** `get_stock_price` fails 12% of the time → Fix Yahoo Finance integration!

---

### **4. The Brain Power Meter (Token Usage = Cost!)**
```python
LLM_TOKEN_USAGE = Counter('financial_rag_llm_tokens_total',
                         'LLM token usage',
                         ['type'])
```
**Tokens = AI's "thinking units" = MONEY!**
- OpenAI charges per 1,000 tokens
- **`type` labels:** "prompt" (input) vs "completion" (output)

**Example data:**
```
financial_rag_llm_tokens_total{type="prompt"} 125,400
financial_rag_llm_tokens_total{type="completion"} 89,200
```

**Cost calculation:**
```
Total tokens = 125,400 + 89,200 = 214,600
Cost (GPT-3.5) = 214,600 ÷ 1000 × $0.002 = $0.43
Average per query = $0.43 ÷ 150 queries = $0.0029
```

**Business critical:** Know your costs!

---

### **5. The Knowledge Base Size (Vector Store)**
```python
VECTOR_STORE_SIZE = Gauge('financial_rag_vector_store_documents',
                         'Number of documents in vector store')
```
**Gauge = Goes up and down** (not just increments)
- Current number of documents
- Updated when documents added/removed

**Example data:**
```
financial_rag_vector_store_documents 1,542
```

**Monitoring:**
- Growing over time? Good! (Adding knowledge)
- Suddenly drops? Bad! (Data loss!)
- Expected growth rate: +50 documents/day

---

## **🎭 The MetricsCollector Class**

### **Recording a Query:**
```python
def record_query(self, status: str, agent_type: str, duration: float):
    QUERY_COUNTER.labels(status=status, agent_type=agent_type).inc()
    QUERY_DURATION.observe(duration)
```

**Usage in our agent:**
```python
start_time = time.time()
try:
    answer = agent.analyze("What's Apple's stock price?")
    duration = time.time() - start_time
    metrics_collector.record_query("success", "tool_using_agent", duration)
except:
    duration = time.time() - start_time
    metrics_collector.record_query("failure", "tool_using_agent", duration)
```

### **Recording Tool Usage:**
```python
def record_tool_usage(self, tool_name: str, success: bool):
    status = "success" if success else "failure"
    AGENT_TOOL_USAGE.labels(tool_name=tool_name, status=status).inc()
```

**Usage when agent uses a tool:**
```python
try:
    price = FinancialTools.get_stock_price("AAPL")
    metrics_collector.record_tool_usage("get_stock_price", True)
except:
    metrics_collector.record_tool_usage("get_stock_price", False)
```

---

## **📊 Prometheus: The Metrics Database**

### **What is Prometheus?**
- Industry-standard monitoring system
- **Pulls metrics** from applications
- **Time-series database** (stores metrics over time)
- **Alerting** when thresholds crossed
- **Grafana** for beautiful dashboards

### **Exposing Metrics:**
```python
def get_metrics(self):
    return generate_latest()
```

**FastAPI endpoint:**
```python
@app.get("/metrics")
def get_metrics():
    return Response(metrics_collector.get_metrics(), 
                   media_type="text/plain")
```

**Prometheus scrapes every 15 seconds:**
```
Prometheus → GET http://ourapp:8000/metrics
← Returns all metrics in text format
← Stores in time-series database
```

---

## **🎓 Real-World Scenarios**

### **Scenario 1: Cost Overrun Alert**
```
Monday: Normal usage, $2.50/day
Tuesday: Bug causes infinite loops!
Metrics show: token_usage{type="prompt"} spikes 1000×
Alert: "Token usage 1000% above normal!"
Team fixes bug before huge bill!
```

### **Scenario 2: Performance Degradation**
```
Week 1: 95% of queries < 1 second
Week 2: Only 70% of queries < 1 second
Investigation: Vector store grew 10×
Solution: Optimize search or scale resources
```

### **Scenario 3: Tool Reliability**
```
Tool "get_stock_price": Success rate 88%
Tool "search_filings": Success rate 99%
Focus improvement efforts on stock price API!
```

---

## **💡 Classroom Activities**

### **Activity 1: The Dashboard Design**
```python
# Task: "Design metrics for a new feature"
# Students propose:
# 1. What to measure?
# 2. What type of metric? (Counter, Gauge, Histogram)
# 3. What labels?
# 4. What insights would it provide?

# Example: "Add user satisfaction metric"
USER_SATISFACTION = Gauge('financial_rag_user_satisfaction',
                         'User satisfaction score (1-5)')
```

### **Activity 2: The Alert Challenge**
```python
# Task: "Create alert rules"
# Given metrics, students write alert conditions:

# Alert 1: If success rate < 90% for 5 minutes
# Alert 2: If average query time > 5 seconds  
# Alert 3: If cost > $10/hour
# Alert 4: If vector store loses > 10% documents
```

### **Activity 3: The Cost Calculator**
```python
# Task: "Calculate business costs"
# Given:
# - 10,000 queries/day
# - Average 800 tokens/query
# - GPT-3.5 Turbo cost: $0.002/1K tokens
# - Server cost: $200/month
# - Developer salary: $120,000/year

# Calculate: Monthly total cost, cost per query, break-even pricing
```

---

## **⚡ Pro Metrics Tips**

### **1. Add Business Metrics:**
```python
# Revenue tracking (if you charge users)
REVENUE = Counter('financial_rag_revenue_total', 'Total revenue')
USER_COUNT = Gauge('financial_rag_active_users', 'Active users')
```

### **2. Add Quality Metrics:**
```python
# Answer quality (human-rated)
ANSWER_QUALITY = Histogram('financial_rag_answer_quality',
                          'Human-rated answer quality (1-5)')
CITATION_ACCURACY = Gauge('financial_rag_citation_accuracy',
                         'Percentage of claims with correct citations')
```

### **3. Add SLOs (Service Level Objectives):**
```python
# Define targets
SLO_AVAILABILITY = 0.999  # 99.9% uptime
SLO_LATENCY = 2.0        # 95% of queries < 2 seconds
SLO_SUCCESS_RATE = 0.98  # 98% success rate
```

---

## **🔍 The Magic of Labels**

### **Why Labels Are Powerful:**
```python
# Without labels:
queries_total 150  # Just a number

# With labels:
queries_total{status="success", agent_type="tool", user_type="premium"} 120
queries_total{status="success", agent_type="tool", user_type="free"} 25
queries_total{status="failure", agent_type="simple", user_type="free"} 5
```

**Now you can analyze:**
- Premium vs free users
- Tool agent vs simple agent
- Success rates by user type

---

## **🎯 Key Takeaways**

1. **Metrics = Visibility** - Know what's happening inside your AI
2. **Counters track events** - Queries, tool usage, tokens
3. **Histograms track distributions** - Response times, quality scores
4. **Gauges track current state** - Vector store size, active users
5. **Labels enable slicing** - Analyze by type, user, feature

**This transforms our AI from:**
- **"Black box"** → **"Glass box"**
- **"Guess about performance"** → **"Data-driven decisions"**
- **"Reactive firefighting"** → **"Proactive optimization"**

**Question for discussion:** If you could add one more metric to track about our Financial AI, what would it be and why?

Create `src/financial_rag/monitoring/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from loguru import logger
import time

# Metrics for Prometheus
QUERY_COUNTER = Counter('financial_rag_queries_total', 'Total number of queries', ['status', 'agent_type'])
QUERY_DURATION = Histogram('financial_rag_query_duration_seconds', 'Query duration in seconds')
AGENT_TOOL_USAGE = Counter('financial_rag_agent_tool_usage_total', 'Agent tool usage', ['tool_name', 'status'])
VECTOR_STORE_SIZE = Gauge('financial_rag_vector_store_documents', 'Number of documents in vector store')
LLM_TOKEN_USAGE = Counter('financial_rag_llm_tokens_total', 'LLM token usage', ['type'])

class MetricsCollector:
    """Collect and expose metrics for Prometheus"""
    
    def __init__(self):
        self.metrics_registry = {}
    
    def record_query(self, status: str, agent_type: str, duration: float):
        """Record query metrics"""
        QUERY_COUNTER.labels(status=status, agent_type=agent_type).inc()
        QUERY_DURATION.observe(duration)
    
    def record_tool_usage(self, tool_name: str, success: bool):
        """Record agent tool usage"""
        status = "success" if success else "failure"
        AGENT_TOOL_USAGE.labels(tool_name=tool_name, status=status).inc()
    
    def record_token_usage(self, token_type: str, count: int):
        """Record LLM token usage"""
        LLM_TOKEN_USAGE.labels(type=token_type).inc(count)
    
    def update_vector_store_size(self, size: int):
        """Update vector store document count"""
        VECTOR_STORE_SIZE.set(size)
    
    def get_metrics(self):
        """Get all metrics in Prometheus format"""
        return generate_latest()

# Global metrics collector
metrics_collector = MetricsCollector()
```

### Update API to Include Metrics Endpoint

Add to `src/financial_rag/api/server.py`:

```python
from financial_rag.monitoring.metrics import metrics_collector

# Add this route to the FinancialRAGAPI class:
@self.app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response
    return Response(
        content=metrics_collector.get_metrics(),
        media_type="text/plain"
    )

# Update the query_analysis endpoint to record metrics:
@self.app.post("/query", response_model=QueryResponse)
async def query_analysis(request: QueryRequest):
    """Main endpoint for financial analysis"""
    start_time = time.time()
    try:
        # ... existing code ...
        
        # Record metrics
        metrics_collector.record_query(
            status="success",
            agent_type="agent" if request.use_agent else "rag",
            duration=time.time() - start_time
        )
        
        return response
        
    except Exception as e:
        # Record failure metrics
        metrics_collector.record_query(
            status="failure", 
            agent_type="agent" if request.use_agent else "rag",
            duration=time.time() - start_time
        )
        raise
```

## Step 19: Advanced Configuration Management

Create `src/financial_rag/config/__init__.py`

Create `src/financial_rag/config/advanced.py`

# **🎛️ The AI Control Panel: Advanced Configuration Management**

Good morning class! Today we're looking at the **master control panel** for our Financial AI - the advanced configuration system! This is where we set ALL the dials, switches, and knobs that control how our AI behaves!

---

## **🚀 What This Code Does**

This code creates a **smart, validated configuration system** that:
1. **⚙️ Centralizes all settings** in one place
2. **✅ Validates values automatically** (no bad configs!)
3. **🌍 Reads from multiple sources** (.env files, environment variables)
4. **🔧 Provides sensible defaults**
5. **💡 Self-documents** with type hints and descriptions

**Think of it like:** The cockpit of an airplane, with every control labeled, validated, and with safe defaults!

---

## **🎯 Why Advanced Configuration Matters**

### **Without This:**
```python
# Scattered throughout code:
chunk_size = 1000  # In document_processor.py
temperature = 0.1  # In rag_chain.py  
port = 8000        # In server.py
# What if someone sets temperature=5.0? Crashes!
```

### **With This:**
```python
# One source of truth:
config = AdvancedConfig()
config.CHUNK_SIZE        # Always valid (100-2000)
config.LLM_TEMPERATURE   # Always valid (0-1)
config.API_PORT          # Always valid port
# Automatic validation prevents errors!
```

---

## **🔧 The Configuration Sections**

### **Section 1: API Keys (The Secrets)**
```python
OPENAI_API_KEY: str                # Required! No default
WANDB_API_KEY: Optional[str] = None  # Optional
```

**Why this matters:**
- `OPENAI_API_KEY` has **no default** = Must be provided!
- `WANDB_API_KEY` is `Optional` = Can be None
- **Type safety:** Python knows which are required

---

### **Section 2: Model Settings (The AI Brain)**
```python
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local, free
LLM_MODEL: str = "gpt-3.5-turbo"           # Cost-effective
LLM_TEMPERATURE: float = 0.1               # Low = Consistent
LLM_MAX_TOKENS: int = 2000                 # Limit response size
```

**Smart defaults:**
- **`all-MiniLM-L6-v2`** = Local embedding (no API costs!)
- **`gpt-3.5-turbo`** = Cheaper than GPT-4
- **Temperature 0.1** = Very consistent for financial data
- **2000 tokens** = Prevents extremely long (expensive) responses

---

### **Section 3: RAG Settings (The Memory System)**
```python
CHUNK_SIZE: int = 1000      # Optimal for financial docs
CHUNK_OVERLAP: int = 200    # Prevent cutting sentences
TOP_K_RESULTS: int = 3      # Top 3 relevant chunks
SEARCH_TYPE: str = "similarity"  # "similarity" or "mmr"
```

**Research-backed defaults:**
- **1000 chars/chunk** = Fits most financial paragraphs
- **20% overlap** = Ensures context continuity
- **Top 3 results** = Balance of coverage vs. token usage

---

### **Section 4: Agent Settings (The Thinking Engine)**
```python
AGENT_MAX_ITERATIONS: int = 5           # Prevent infinite loops
AGENT_ENABLE_MONITORING: bool = True    # Always monitor in production
```

**Safety first:**
- **Max 5 iterations** = Agent can't get stuck thinking forever
- **Monitoring enabled** = Track everything for debugging

---

### **Section 5: API Settings (The Delivery System)**
```python
API_HOST: str = "0.0.0.0"    # Accept connections from anywhere
API_PORT: int = 8000         # Standard web port
API_WORKERS: int = 1         # Development = 1, Production = 4+
API_LOG_LEVEL: str = "info"  # Not too noisy, not too quiet
```

**Production-ready:**
- **`0.0.0.0`** = Critical for Docker/Kubernetes
- **Port 8000** = Standard for Python web apps
- **Workers** = Scale based on CPU cores

---

### **Section 6: Storage Settings (The Filing Cabinet)**
```python
VECTOR_STORE_PATH: str = "./data/chroma_db"
RAW_DATA_PATH: str = "./data/raw" 
PROCESSED_DATA_PATH: str = "./data/processed"
```

**Organized structure:**
```
data/
├── raw/           # Original SEC filings
├── processed/     # Cleaned documents  
└── chroma_db/     # Vector database
```

---

### **Section 7: Kubernetes Settings (The Cloud Factory)**
```python
K8S_NAMESPACE: str = "financial-rag"
K8S_DEPLOYMENT_NAME: str = "financial-rag-api"
```

**Infrastructure as code:** Names match our Kubernetes manifests!

---

### **Section 8: Monitoring Settings (The Observability)**
```python
PROMETHEUS_ENABLED: bool = True  # Always track metrics
WANDB_ENABLED: bool = True       # Experiment tracking
```

**Observability by default:** You can't fix what you can't measure!

---

## **✅ The Magic: Automatic Validation**

### **Validator 1: Chunk Size**
```python
@validator("CHUNK_SIZE")
def validate_chunk_size(cls, v):
    if v < 100 or v > 2000:
        raise ValueError("CHUNK_SIZE must be between 100 and 2000")
    return v
```

**Why validate?**
- **Too small (<100):** Chunks lose context
- **Too large (>2000):** Won't fit in AI context window
- **Automatic check:** Prevents deployment with bad config!

### **Validator 2: Temperature**
```python
@validator("LLM_TEMPERATURE")
def validate_temperature(cls, v):
    if v < 0 or v > 1:
        raise ValueError("LLM_TEMPERATURE must be between 0 and 1")
    return v
```

**Temperature range:**
- **0.0** = Completely deterministic (always same answer)
- **0.5** = Balanced creativity
- **1.0** = Very creative (might invent financial data!)
- **We use 0.1** = Slight variation, mostly consistent

### **Validator 3: Top K Results**
```python
@validator("TOP_K_RESULTS")
def validate_top_k(cls, v):
    if v < 1 or v > 10:
        raise ValueError("TOP_K_RESULTS must be between 1 and 10")
    return v
```

**Search optimization:**
- **1 result** = Might miss relevant info
- **10 results** = Too many, expensive, noisy
- **3 results** = Sweet spot (research-backed)

---

## **🌍 The Configuration Sources**

### **Priority Order:**
```python
class Config:
    env_file = ".env"        # 1. Read from .env file
    case_sensitive = False   # 2. Case-insensitive (OPENAI_API_KEY = openai_api_key)
```

**What Pydantic checks (in order):**
1. **Arguments passed** to `AdvancedConfig(...)`
2. **Environment variables** (system/process)
3. **`.env` file** in current directory
4. **Default values** in the class

**Example loading:**
```python
# If .env contains:
# OPENAI_API_KEY=sk-abc123
# CHUNK_SIZE=1500

config = AdvancedConfig()
config.OPENAI_API_KEY  # "sk-abc123" (from .env)
config.CHUNK_SIZE      # 1500 (from .env, not default 1000)
config.API_PORT        # 8000 (default, not in .env)
```

---

## **🎓 Real-World Usage Examples**

### **Example 1: Development vs Production**
```python
# Development (.env.development):
OPENAI_API_KEY=sk-dev-key
LLM_MODEL=gpt-3.5-turbo
API_WORKERS=1
LOG_LEVEL=debug

# Production (.env.production):
OPENAI_API_KEY=sk-prod-key  
LLM_MODEL=gpt-4
API_WORKERS=4
LOG_LEVEL=warning
# Same code, different configs!
```

### **Example 2: Environment Variables (Cloud)**
```bash
# Heroku/Render set environment variables:
export OPENAI_API_KEY=sk-abc123
export API_WORKERS=4
export CHUNK_SIZE=1200

# Our app automatically uses them!
python app.py
```

### **Example 3: Validation Protection**
```python
# What happens with bad config:
try:
    config = AdvancedConfig(CHUNK_SIZE=5000)
except ValueError as e:
    print(e)  # "CHUNK_SIZE must be between 100 and 2000"
# Prevents deployment with bad settings!
```

---

## **💡 Classroom Activities**

### **Activity 1: The Configuration Detective**
```python
# Task: "Find optimal settings"
# Students experiment:
config1 = AdvancedConfig(LLM_TEMPERATURE=0.0)
config2 = AdvancedConfig(LLM_TEMPERATURE=0.5)  
config3 = AdvancedConfig(LLM_TEMPERATURE=1.0)

# Compare AI responses: Which is best for financial analysis?
```

### **Activity 2: The Validation Challenge**
```python
# Task: "Add a new validator"
# Students add validation for:
# 1. API_PORT (1024-65535)
# 2. CHUNK_OVERLAP (must be < CHUNK_SIZE)
# 3. LLM_MAX_TOKENS (100-4000)

@validator("API_PORT")
def validate_port(cls, v):
    if v < 1024 or v > 65535:
        raise ValueError("API_PORT must be between 1024 and 65535")
    return v
```

### **Activity 3: The Cost Optimization**
```python
# Task: "Reduce costs 50%"
# Given: 10,000 queries/day, current cost $100/day
# Students adjust config to reduce cost:

config = AdvancedConfig(
    LLM_MODEL="gpt-3.5-turbo",  # Was gpt-4
    TOP_K_RESULTS=2,             # Was 3
    LLM_MAX_TOKENS=1000          # Was 2000
)
# Calculate new estimated cost!
```

---

## **⚡ Pro Configuration Tips**

### **1. Add Config Profiles:**
```python
# Could extend to profiles:
class DevelopmentConfig(AdvancedConfig):
    API_WORKERS = 1
    LOG_LEVEL = "debug"
    
class ProductionConfig(AdvancedConfig):
    API_WORKERS = 4
    LOG_LEVEL = "warning"
```

### **2. Add Secret Validation:**
```python
@validator("OPENAI_API_KEY")
def validate_openai_key(cls, v):
    if not v.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key format")
    return v
```

### **3. Add Derived Settings:**
```python
@property
def max_chars_per_query(self):
    # Calculate based on other settings
    return self.TOP_K_RESULTS * self.CHUNK_SIZE * 4  # Rough estimate
```

---

## **🔍 The Magic of Pydantic Settings**

### **Automatic Type Conversion:**
```python
# .env file has strings:
# API_PORT="8000"
# LLM_TEMPERATURE="0.1"

# Pydantic converts automatically:
config.API_PORT           # int 8000 (not str "8000")
config.LLM_TEMPERATURE    # float 0.1 (not str "0.1")
```

### **Case Insensitive:**
```python
# All these work:
export openai_api_key=sk-abc123
export OPENAI_API_KEY=sk-abc123  
export Openai_Api_Key=sk-abc123
# config.OPENAI_API_KEY gets them all!
```

### **Dotenv Support:**
```python
# .env file format:
OPENAI_API_KEY=sk-abc123
# Optional comments
WANDB_API_KEY=wandb_xyz  # For experiment tracking
# Multi-line values
LONG_DESCRIPTION="This is a \
multi-line description"
```

---

## **🎯 Key Takeaways**

1. **Centralized config** = Single source of truth
2. **Automatic validation** = Prevents deployment errors
3. **Sensible defaults** = Good out-of-the-box experience
4. **Multiple sources** = Flexible deployment options
5. **Type safety** = Python knows what to expect

**This transforms configuration from:**
- **"Magic strings scattered everywhere"** → **"Validated, typed configuration"**
- **"Runtime errors"** → **"Startup validation errors"**
- **"Environment-specific hacks"** → **"Profiles and overrides"**

**Question for discussion:** If you were to add one more configuration setting to make our Financial AI even more powerful, what would it be and why?

```python
import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, validator
from loguru import logger

class AdvancedConfig(BaseSettings):
    """Advanced configuration with validation"""
    
    # API Settings
    OPENAI_API_KEY: str
    WANDB_API_KEY: Optional[str] = None
    
    # Model Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 3
    SEARCH_TYPE: str = "similarity"  # "similarity" or "mmr"
    
    # Agent Settings
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_ENABLE_MONITORING: bool = True
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_LOG_LEVEL: str = "info"
    
    # Storage Settings
    VECTOR_STORE_PATH: str = "./data/chroma_db"
    RAW_DATA_PATH: str = "./data/raw"
    PROCESSED_DATA_PATH: str = "./data/processed"
    
    # Kubernetes Settings
    K8S_NAMESPACE: str = "financial-rag"
    K8S_DEPLOYMENT_NAME: str = "financial-rag-api"
    
    # Monitoring Settings
    PROMETHEUS_ENABLED: bool = True
    WANDB_ENABLED: bool = True
    
    @validator("CHUNK_SIZE")
    def validate_chunk_size(cls, v):
        if v < 100 or v > 2000:
            raise ValueError("CHUNK_SIZE must be between 100 and 2000")
        return v
    
    @validator("LLM_TEMPERATURE") 
    def validate_temperature(cls, v):
        if v < 0 or v > 1:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 1")
        return v
    
    @validator("TOP_K_RESULTS")
    def validate_top_k(cls, v):
        if v < 1 or v > 10:
            raise ValueError("TOP_K_RESULTS must be between 1 and 10")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global advanced config
advanced_config = AdvancedConfig()
```

## Step 20: Final Production Scripts

Create `scripts/deploy.sh`

```bash
#!/bin/bash

set -e

echo "🚀 Deploying Financial RAG Agent to Kubernetes..."

# Validate environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is required"
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t financial-rag-agent:latest .

# If using remote registry, push the image
# docker tag financial-rag-agent:latest your-registry/financial-rag-agent:latest
# docker push your-registry/financial-rag-agent:latest

# Create namespace if it doesn't exist
echo "📁 Creating Kubernetes namespace..."
kubectl apply -f kubernetes/namespace.yaml

# Create secrets
echo "🔐 Creating secrets..."
kubectl create secret generic financial-rag-secrets \
    --namespace=financial-rag \
    --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" \
    --from-literal=WANDB_API_KEY="$WANDB_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply all Kubernetes manifests
echo "📄 Applying Kubernetes manifests..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/persistent-volume-claim.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/financial-rag-api -n financial-rag --timeout=300s

# Get service information
echo "🌐 Service information:"
kubectl get service -n financial-rag

echo "✅ Deployment completed successfully!"
echo "📊 Check logs: kubectl logs -f deployment/financial-rag-api -n financial-rag"
echo "🌐 Access API: kubectl port-forward service/financial-rag-service 8000:8000 -n financial-rag"
```

Create `scripts/health-check.sh`

```bash
#!/bin/bash

set -e

echo "🏥 Running comprehensive health check..."

NAMESPACE=${1:-financial-rag}
SERVICE=${2:-financial-rag-service}
PORT=${3:-8000}

# Check if namespace exists
echo "1. Checking namespace..."
kubectl get namespace $NAMESPACE > /dev/null 2>&1 || {
    echo "❌ Namespace $NAMESPACE does not exist"
    exit 1
}

# Check deployment status
echo "2. Checking deployment..."
DEPLOYMENT_STATUS=$(kubectl get deployment financial-rag-api -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')
if [ "$DEPLOYMENT_STATUS" != "True" ]; then
    echo "❌ Deployment not available"
    exit 1
fi

# Check pod status
echo "3. Checking pods..."
POD_READY=$(kubectl get pods -n $NAMESPACE -l app=financial-rag-api -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')
if [ "$POD_READY" != "True" ]; then
    echo "❌ Pod not ready"
    exit 1
fi

# Port forward and test API
echo "4. Testing API health endpoint..."
kubectl port-forward service/$SERVICE $PORT:$PORT -n $NAMESPACE > /dev/null 2>&1 &
PORT_FORWARD_PID=$!

# Wait for port forward to be established
sleep 5

# Test health endpoint
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health || true)

# Kill port forward
kill $PORT_FORWARD_PID > /dev/null 2>&1 || true

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Health check passed - API is responding"
else
    echo "❌ Health check failed - API returned HTTP $HEALTH_RESPONSE"
    exit 1
fi

echo "🎉 All health checks passed! System is operational."
```

Create `kubernetes/kustomization.yaml`

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: financial-rag

resources:
  - namespace.yaml
  - configmap.yaml
  - secret.yaml
  - persistent-volume-claim.yaml
  - deployment.yaml
  - service.yaml
  - hpa.yaml
  - ingress.yaml

commonLabels:
  app: financial-rag-api
  version: v1

images:
  - name: financial-rag-agent
    newTag: latest
```

By now, our Project Structure should look like this

```
financial-rag-agent/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── kubernetes/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   ├── persistent-volume-claim.yaml
│   └── kustomization.yaml
├── scripts/
│   ├── deploy.sh
│   ├── health-check.sh
│   ├── start_api.py
│   └── test_production.py
├── src/
│   └── financial_rag/
│       ├── config/
│       │   ├── __init__.py
│       │   └── advanced.py
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── tracing.py
│       │   └── metrics.py
│       └── api/
│           ├── __init__.py
│           ├── models.py
│           └── server.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env
├── .dockerignore
└── README.md
```

## Step 22: Final Deployment Commands

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Deploy to Kubernetes
./scripts/deploy.sh

# Run health check
./scripts/health-check.sh

# Or deploy with kustomize
kubectl apply -k kubernetes/

# Check status
kubectl get all -n financial-rag

# View logs
kubectl logs -f deployment/financial-rag-api -n financial-rag

# Port forward for local access
kubectl port-forward service/financial-rag-service 8000:8000 -n financial-rag
```

## 🎉 Enterprise Deployment Complete!

Your Financial RAG Analyst Agent is now fully enterprise-ready with:

### ✅ **Production Features:**
- **Kubernetes Deployment** with auto-scaling
- **CI/CD Pipeline** with GitHub Actions
- **Monitoring & Metrics** with Prometheus/WandB
- **Health Checks** and readiness probes
- **Config Management** with ConfigMaps and Secrets
- **Persistent Storage** for vector database
- **Load Balancing** and ingress
- **Security** with non-root containers

### ✅ **AI/ML Capabilities:**
- **Intelligent Agent** with tool usage
- **RAG System** with sophisticated chunking
- **Real-time Data** integration
- **Multiple Analysis Styles** (analyst, executive, risk)
- **Comprehensive Monitoring** of AI components

### ✅ **DevOps Excellence:**
- **Containerized** with Docker
- **Orchestrated** with Kubernetes
- **Automated Deployment** with CI/CD
- **Infrastructure as Code**
- **Production-grade** configuration

### 🚀 **Ready for Production Use:**

```bash
# Final verification
python test_production.py
./scripts/health-check.sh

# Access the API
kubectl port-forward service/financial-rag-service 8000:8000 -n financial-rag
# Visit: http://localhost:8000/docs
```

Your Financial RAG Analyst Agent is now a robust, scalable, enterprise-grade system that demonstrates your full-stack AI engineering capabilities - from backend DevOps to sophisticated AI agent systems!


