# Docker Deployment

Deploy Supreme System V5 using Docker Compose.

## Demo Environment

```bash
docker-compose -f docker-compose.demo.yml up -d
```

## Production Deployment

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Services

- **supreme-system**: Main trading platform
- **prometheus**: Metrics collection
- **grafana**: Visualization dashboards

## Health Checks

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f supreme-system

# Restart services
docker-compose restart
```

