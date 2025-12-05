
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












Good morning class! Today we're looking at the **most advanced and professional** part of software engineering - the CI/CD pipeline! This is like having a **fully automated AI factory** that builds, tests, and deploys our Financial Agent every time we make a change!


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

## **📋 Trigger: When Does This Run?**

```yaml
on:
  push:
    branches: [ main, develop ]  # Code push to main or develop
  pull_request:
    branches: [ main ]          # PR to main branch
```

**Two triggers:**
1. **Push to main/develop** → "Deploy new features"
2. **Pull request to main** → "Test before merging"

**Why both?**
- **PRs:** Catch bugs BEFORE they reach main
- **Push to develop:** Test integration
- **Push to main:** Deploy to production

---

## **🎭 The Environment Variables**

```yaml
env:
  IMAGE_NAME: financial-rag-agent
  REGISTRY: ghcr.io  # GitHub Container Registry
```

**Shared across all jobs:**
- `IMAGE_NAME` = Our Docker image name
- `REGISTRY` = Where to store images (GitHub)
- **Consistency:** Same values everywhere

---

## **🧪 Job 1: The Quality Control (Test)**

### **Setup:**
```yaml
runs-on: ubuntu-latest
strategy:
  matrix:
    python-version: [3.9]  # Could add [3.9, 3.10, 3.11]
```
- **Ubuntu Linux** runner
- **Matrix testing:** Test on multiple Python versions
- **Future:** Add 3.10, 3.11 for compatibility

### **Step 1: Checkout Code**
```yaml
- uses: actions/checkout@v4
```
- **Git clone** our repository
- Uses GitHub's optimized checkout action
- **Version @v4** = Latest stable

### **Step 2: Python Setup**
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: ${{ matrix.python-version }}
```
- Installs **Python 3.9**
- Uses GitHub's pre-built Python
- **Fast:** No compiling Python from source

### **Step 3: Install Dependencies**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -e .[dev]
```
**Two commands:**
1. **Upgrade pip** (always latest)
2. **Install our package** with `[dev]` extras
   - `-e .` = "Editable" install (links to code)
   - `[dev]` = Development dependencies (pytest, etc.)

### **Step 4: Run Tests**
```yaml
- name: Run tests
  run: |
    python test_foundation.py
    python test_agent.py
    python test_production.py
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```
**Three test suites:**
1. `test_foundation.py` = Core components work
2. `test_agent.py` = AI agent works
3. `test_production.py` = Production readiness

**Secret environment variable:**
- `${{ secrets.OPENAI_API_KEY }}` = From GitHub Secrets
- **Never in code!** Secure storage
- Tests need real API key

### **Step 5: Security Scan**
```yaml
- name: Run security scan
  run: |
    pip install bandit safety
    bandit -r src/ -f json -o bandit-report.json
    safety check --json
```
**Two security tools:**
1. **Bandit** = Python code security scanner
   - Finds hardcoded passwords, SQL injection risks
2. **Safety** = Dependency vulnerability scanner
   - Checks if any packages have known security issues

**Professional practice:** Security scanning in CI!

---

## **🏗️ Job 2: The Factory Floor (Build & Push)**

### **Dependency:**
```yaml
needs: test
if: github.ref == 'refs/heads/main'
```
- **`needs: test`** = Only runs if tests pass!
- **`if: main`** = Only on main branch (not PRs or develop)

### **Step 1: Docker Login**
```yaml
- name: Log in to GitHub Container Registry
  uses: docker/login-action@v2
  with:
    registry: ${{ env.REGISTRY }}
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}
```
**Authenticates to GitHub Container Registry:**
- `${{ github.actor }}` = Who triggered the workflow
- `${{ secrets.GITHUB_TOKEN }}` = Automatic GitHub token
- **Secure:** No manual passwords needed

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
**Automatic tagging magic:**
- **Branch-based:** `main` → `latest`
- **PR-based:** `pr-123` → `pr-123`
- **Semantic version:** `v1.2.3` → `1.2.3`, `1.2`, `1`
- **Commit-based:** `main-abc123def`

**Example tags created:**
```
ghcr.io/yourname/financial-rag-agent:latest
ghcr.io/yourname/financial-rag-agent:main
ghcr.io/yourname/financial-rag-agent:1.2.3
ghcr.io/yourname/financial-rag-agent:1.2
ghcr.io/yourname/financial-rag-agent:1
ghcr.io/yourname/financial-rag-agent:main-abc123def
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
```
**Smart building:**
- `cache-from: type=gha` = Use GitHub Actions cache
- **Fast builds:** Layers cached between runs
- `push: true` = Push to registry automatically
- `labels` = Metadata about build (git sha, date, etc.)

---

## **🚀 Job 3: The Showroom (Staging Deployment)**

### **Dependencies:**
```yaml
needs: build-and-push
environment: staging
```
- **`needs: build-and-push`** = After successful build
- **`environment: staging`** = GitHub environment protection
  - Can require approvals
  - Separate secrets
  - Audit log

### **Step 1: Deploy to Kubernetes**
```yaml
- name: Deploy to Kubernetes
  uses: steebchen/kubectl@v2
  with:
    config: ${{ secrets.KUBECONFIG_STAGING }}
    command: apply -f kubernetes/
```
**What happens:**
1. Uses `kubectl` GitHub Action
2. Authenticates with `KUBECONFIG_STAGING` secret
3. Runs `kubectl apply -f kubernetes/`
4. **Applies ALL Kubernetes manifests** from `kubernetes/` folder

### **Step 2: Verify Deployment**
```yaml
- name: Verify deployment
  uses: steebchen/kubectl@v2
  with:
    config: ${{ secrets.KUBECONFIG_STAGING }}
    command: rollout status deployment/financial-rag-api
```
**Waits for:**
- Pods to start
- Health checks to pass
- Service to be ready
- **If fails:** Pipeline fails (prevents bad deployments!)

---

## **🌍 Job 4: Customer Delivery (Production)**

### **The Gatekeeper:**
```yaml
needs: deploy-staging
environment: production
if: github.ref == 'refs/heads/main'
```
**Three safety checks:**
1. **`needs: deploy-staging`** = Staging must pass first
2. **`environment: production`** = Extra protection layer
3. **`if: main`** = Only from main branch

**Production environment typically requires:**
- Manual approval
- Multiple reviewers
- Business hours only

### **Same steps as staging:**
- Deploy to production Kubernetes
- Verify deployment
- **But with `KUBECONFIG_PRODUCTION` secret!**

---

## **🎓 Real-World Scenarios**

### **Scenario 1: Developer Makes a Change**
```
1. Developer pushes to "develop" branch
2. Pipeline runs tests ONLY (no deploy)
3. Tests pass → Developer creates PR to main
4. Pipeline runs tests on PR
5. Team reviews, approves, merges
6. Pipeline: Test → Build → Deploy staging → Deploy production
7. New feature live in 10 minutes!
```

### **Scenario 2: Security Vulnerability Found**
```
1. Safety scan finds vulnerable dependency
2. Pipeline FAILS at test job
3. Developer gets notification
4. Fix dependency, push again
5. Pipeline passes, safe deployment continues
```

### **Scenario 3: Rollback Needed**
```
1. Production deployment has bug
2. Developer: git revert last commit
3. Push to main
4. Pipeline builds OLD working version
5. Deploys it automatically
6. Bug fixed in 5 minutes!
```

---

## **🔍 The Magic of GitHub Actions**

### **Secrets Management:**
```yaml
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

---

## **⚡ Pro Pipeline Tips**

### **1. Add Notifications:**
```yaml
- name: Send Slack notification
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#alerts'
```

### **2. Add Performance Tests:**
```yaml
- name: Run performance tests
  run: |
    pip install locust
    locust -f performance_test.py --headless -u 100 -r 10 -t 1m
```

### **3. Add Cost Estimation:**
```yaml
- name: Estimate deployment cost
  run: |
    python estimate_cost.py
  # Calculates: OpenAI tokens, storage, compute costs
```

---

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

## Step 18: Monitoring and Metrics

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


