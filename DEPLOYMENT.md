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
