# Docker Compose Basics

Docker Compose is a tool for defining and running multi-container applications. You define your services in a `docker-compose.yml` file and manage them with simple commands.

## Key Commands

- `docker compose up -d` — start all services in detached mode
- `docker compose down` — stop and remove all containers
- `docker compose logs -f` — follow logs from all services
- `docker compose ps` — list running services

## Volumes

Named volumes persist data between container restarts. Define them in the `volumes:` section and reference them in service configurations. Data in named volumes survives `docker compose down` but is removed with `docker compose down -v`.

## Networks

Services on the same Docker network can communicate using their service names as hostnames. Use `127.0.0.1` port bindings to restrict access to localhost only.

## Health Checks

Health checks let Docker monitor whether a service is ready. A service with a failing health check will be marked as unhealthy. Dependent services can use `depends_on` with `condition: service_healthy` to wait for readiness.

## GPU Passthrough

For NVIDIA GPU access inside containers, use the `deploy.resources.reservations.devices` configuration with `driver: nvidia` and `capabilities: [gpu]`. Requires the NVIDIA Container Toolkit on the host.
