# Docker Engine Install — WSL2 Design Spec
**Date:** 2026-03-27
**Environment:** WSL2, Ubuntu 24.04 LTS (Noble), user `avik2007` (has sudo)
**Scope:** Install and activate Docker Engine natively in WSL2. No Docker Desktop. No Windows integration.

---

## Goal

Enable `docker build`, `docker run`, and `docker compose` inside WSL2 so the `mhw-risk-profiler` Dockerfile can be built and run locally before cloud deployment.

---

## Approach

Docker Engine (Option A) — native install via Docker's official apt repository. Rootless and Podman approaches were considered and rejected as unnecessary complexity for a single-developer research environment.

---

## Install Steps

1. **Remove conflicting packages** from Ubuntu's default repo:
   - `docker.io`, `docker-doc`, `docker-compose`, `podman-docker`, `containerd`, `runc`

2. **Add Docker's official apt source:**
   - Install prereqs: `ca-certificates`, `curl`
   - Add GPG key to `/etc/apt/keyrings/docker.asc`
   - Add repo: `https://download.docker.com/linux/ubuntu` (Noble / amd64)

3. **Install packages:**
   - `docker-ce`
   - `docker-ce-cli`
   - `containerd.io`
   - `docker-buildx-plugin`
   - `docker-compose-plugin`

4. **Add user to docker group:**
   - `sudo usermod -aG docker avik2007`
   - New group takes effect on next shell session (or via `newgrp docker`)

---

## Daemon Activation

Check whether `systemd` is active in this WSL2 instance:

```bash
ps -p 1 -o comm=
```

- **If output is `systemd`:** `sudo systemctl enable --now docker` — daemon starts automatically on each WSL session.
- **If output is `init` (no systemd):** Add the following guard to `~/.bashrc`:
  ```bash
  if ! pgrep -x dockerd > /dev/null; then sudo service docker start; fi
  ```

---

## Verification Gate

All three must pass before the task is marked done:

| Check | Command | Expected output |
|---|---|---|
| Daemon running | `docker run hello-world` | "Hello from Docker!" message |
| Compose plugin | `docker compose version` | Version string printed |
| Project build | `docker build -t mhw-risk-profiler .` (from project root) | Build completes without error |

---

## Out of Scope

- GCS credential mounting into containers (deferred to ingestion pipeline task)
- Runtime env vars for `GOOGLE_APPLICATION_CREDENTIALS` inside Docker (same deferral)
- CI/CD Docker build configuration
