# 🚀 Deployment Guide - Supreme System V5

## Production Deployment

### Prerequisites

- **Kubernetes cluster** v1.24+ (GKE, EKS, AKS, or self-hosted)
- **ArgoCD** installed for GitOps
- **PostgreSQL** database (v14+)
- **Redis** cache (v7+)
- **Monitoring stack**: Prometheus + Grafana
- **kubectl** CLI configured
- **Git access** to repository

---

## 🤖 Enterprise AI Coverage Optimizer Deployment

### Enterprise CI/CD Pipeline

Supreme System V5 includes enterprise-grade AI optimization with quota-free operation:

#### Multi-API Key Configuration
```bash
# GitHub Secrets / CI/CD Variables
GEMINI_KEYS=key1,key2,key3,key4,key5,key6,key7,key8,key9,key10
OPENAI_API_KEY=sk-your-openai-key
CLAUDE_API_KEY=sk-ant-your-claude-key

# Optimization parameters
TARGET_COVERAGE=85.0
BATCH_SIZE=3
MAX_CONCURRENT_BATCHES=2
```

#### Enterprise CI/CD Workflow
```yaml
# .github/workflows/enterprise-optimization.yml
name: 🚀 Enterprise AI Coverage Optimization

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM
  workflow_dispatch:
    inputs:
      target_coverage:
        description: 'Target coverage percentage'
        default: '85.0'

jobs:
  enterprise-optimization:
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: 🔑 Setup Enterprise Keys
        run: |
          echo "GEMINI_KEYS=${{ secrets.GEMINI_KEYS }}" >> $GITHUB_ENV
          echo "OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}" >> $GITHUB_ENV
          echo "CLAUDE_API_KEY=${{ secrets.CLAUDE_API_KEY }}" >> $GITHUB_ENV

      - name: 📦 Install Dependencies
        run: pip install -r requirements.txt

      - name: 🚀 Run Enterprise Optimizer
        run: |
          python scripts/enterprise_optimizer.py \
            --target-coverage ${{ github.event.inputs.target_coverage || '85.0' }} \
            --batch-size 3 \
            --max-concurrent 2 \
            --max-iterations 10

      - name: 📊 Upload Coverage Report
        uses: actions/upload-artifact@v4
        with:
          name: enterprise-optimization-report
          path: |
            logs/enterprise_optimizer.log
            coverage-reports/

      - name: 📢 Notify Results
        if: always()
        run: |
          # Send results to Slack/Discord
          curl -X POST ${{ secrets.ALERT_WEBHOOK_URL }} \
            -H 'Content-type: application/json' \
            -d '{"text":"Enterprise AI Optimization Complete"}'
```

#### Environment Variables for Production
```bash
# Production environment
export GEMINI_KEYS="prod_key1,prod_key2,prod_key3,..."
export OPENAI_API_KEY="sk-prod-..."
export CLAUDE_API_KEY="sk-ant-prod-..."

# Enterprise settings
export TARGET_COVERAGE=85.0
export BATCH_SIZE=3
export MAX_CONCURRENT_BATCHES=2
export PRODUCTION_MODE=true

# Monitoring
export ALERT_WEBHOOK_URL="https://hooks.slack.com/..."
export QUOTA_DASHBOARD_URL="https://grafana.company.com/d/quota"
```

#### Key Management Strategy
1. **Separate projects**: Use different Google Cloud projects for key isolation
2. **Quota monitoring**: Track usage across all keys in real-time
3. **Auto-scaling**: Add new keys when quota utilization >80%
4. **Key rotation**: Rotate keys every 30 days for security

#### Enterprise Monitoring Setup
```yaml
# Prometheus metrics for enterprise optimizer
groups:
  - name: enterprise_optimizer
    rules:
    - alert: HighQuotaUsage
      expr: enterprise_quota_usage > 0.8
      labels:
        severity: warning
      annotations:
        summary: "High API quota usage detected"

    - alert: KeyErrorsHigh
      expr: enterprise_key_errors_total > 5
      labels:
        severity: critical
      annotations:
        summary: "Multiple API key errors detected"
```

---

## Quick Deploy

### Option 1: ArgoCD (Recommended)

```bash
# 1. Apply ArgoCD application
kubectl apply -f prod/argo-application.yaml

# 2. Wait for sync
kubectl wait --for=condition=Synced application/supreme-system-v5 -n argocd

# 3. Verify deployment
kubectl get pods -n trading-prod
```

### Option 2: Direct kubectl

```bash
# 1. Create namespace
kubectl create namespace trading-prod

# 2. Configure secrets
kubectl create secret generic supreme-secrets \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  --from-literal=gemini-api-key=AIzaSyBH8mRSlNVKQoRi5uCrEJikTJlqhRhPA-g \
  -n trading-prod

# 3. Deploy application
kubectl apply -f prod/deployment.yaml

# 4. Verify
kubectl get all -n trading-prod
```

---

## Configuration

### Secrets Management

```bash
# Generate JWT secret
JWT_SECRET=$(openssl rand -base64 32)

# Create Kubernetes secret
kubectl create secret generic supreme-secrets \
  --from-literal=jwt-secret=$JWT_SECRET \
  --from-literal=gemini-api-key=YOUR_GEMINI_KEY \
  --from-literal=database-url=postgresql://... \
  --from-literal=redis-url=redis://... \
  -n trading-prod
```

### Environment Configuration

Edit `prod/deployment.yaml` ConfigMap:

```yaml
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"  # DEBUG for troubleshooting
  TARGET_COVERAGE: "80"
  QUANTUM_CRYPTO_ENABLED: "true"
  ZERO_TRUST_ENABLED: "true"
```

---

## Health Checks

### Endpoints

- **Liveness**: `GET /health/live` - Is service alive?
- **Readiness**: `GET /health/ready` - Is service ready for traffic?
- **Startup**: `GET /health/startup` - Has service started?
- **Metrics**: `GET /metrics` - Prometheus metrics

### Testing Health

```bash
# Port forward
kubectl port-forward svc/supreme-system-v5 8000:8000 -n trading-prod

# Test endpoints
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/metrics
```

---

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment supreme-system-v5 --replicas=5 -n trading-prod
```

### Auto-Scaling (HPA)

Horizontal Pod Autoscaler already configured:
- **Min replicas**: 3
- **Max replicas**: 10
- **CPU target**: 70%
- **Memory target**: 80%

```bash
# View HPA status
kubectl get hpa supreme-system-hpa -n trading-prod
```

---

## Monitoring

### Prometheus Metrics

```bash
# Port forward to Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# Open in browser
open http://localhost:9090
```

### Grafana Dashboards

```bash
# Port forward to Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Open in browser
open http://localhost:3000
```

---

## Rollback

### ArgoCD Rollback

```bash
# Rollback to previous version
argocd app rollback supreme-system-v5

# Rollback to specific revision
argocd app rollback supreme-system-v5 --revision 5
```

### kubectl Rollback

```bash
# View rollout history
kubectl rollout history deployment/supreme-system-v5 -n trading-prod

# Rollback to previous version
kubectl rollout undo deployment/supreme-system-v5 -n trading-prod

# Rollback to specific revision
kubectl rollout undo deployment/supreme-system-v5 --to-revision=3 -n trading-prod
```

---

## Troubleshooting

### View Logs

```bash
# Stream logs from all pods
kubectl logs -f -l app=supreme-system-v5 -n trading-prod

# Logs from specific pod
kubectl logs supreme-system-v5-xxxxx -n trading-prod

# Previous container logs (if crashed)
kubectl logs supreme-system-v5-xxxxx -n trading-prod --previous
```

### Debug Pod

```bash
# Exec into pod
kubectl exec -it supreme-system-v5-xxxxx -n trading-prod -- /bin/bash

# Check environment
env | grep -E '(JWT|GEMINI|DATABASE)'

# Test connectivity
curl http://localhost:8000/health/live
```

### Common Issues

#### Pod CrashLoopBackOff
```bash
# Check events
kubectl describe pod supreme-system-v5-xxxxx -n trading-prod

# Check logs
kubectl logs supreme-system-v5-xxxxx -n trading-prod --previous
```

#### Service Not Accessible
```bash
# Check service
kubectl get svc supreme-system-v5 -n trading-prod

# Check endpoints
kubectl get endpoints supreme-system-v5 -n trading-prod
```

---

## Security Hardening

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: supreme-system-network-policy
  namespace: trading-prod
spec:
  podSelector:
    matchLabels:
      app: supreme-system-v5
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system  # Allow from service mesh
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

### Pod Security Standards

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: supreme-system-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  readOnlyRootFilesystem: true
```

---

## Disaster Recovery

### Backup Strategy

```bash
# Backup configuration
kubectl get all -n trading-prod -o yaml > backup/k8s-config.yaml

# Backup secrets (encrypted)
kubectl get secrets -n trading-prod -o yaml | \
  kubeseal -o yaml > backup/sealed-secrets.yaml
```

### Recovery Procedure

1. Restore namespace
2. Apply secrets
3. Apply configuration
4. Verify health
5. Resume traffic

---

**Last Updated**: November 12, 2025  
**Maintained by**: DevOps Team  
**Classification**: Internal
