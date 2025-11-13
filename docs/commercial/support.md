# Support

Get help with Supreme System V5.

## Support Channels

- **Email**: thanhmuefatty07@gmail.com
- **Response Time**: 
  - Evaluation: 5 business days
  - Commercial: 24 hours
  - Enterprise: 4 hours

## Documentation

- [Quick Start Guide](../getting-started/quickstart.md)
- [Architecture Overview](../architecture/overview.md)
- [API Reference](../api/trading.md)

## Common Issues

### Dashboard not loading

Check Docker logs:
```bash
docker-compose logs supreme-system
```

### Metrics missing

Verify Prometheus is running:
```bash
docker-compose ps prometheus
```

