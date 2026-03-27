# Docker Engine Install (WSL2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install Docker Engine natively on WSL2 Ubuntu 24.04 and verify it can build the `mhw-risk-profiler` Dockerfile.

**Architecture:** Install via Docker's official apt repo, enable the daemon through systemd (which is confirmed active), and add the user to the `docker` group to avoid requiring `sudo` per command.

**Tech Stack:** Ubuntu 24.04 (Noble), apt, systemd, Docker Engine CE, docker-compose-plugin.

---

### Task 1: Remove conflicting packages

**Files:**
- No files modified — system package state only.

- [ ] **Step 1: Purge Ubuntu-repo Docker packages**

These packages conflict with Docker's official CE packages. This is safe even if none are installed — apt will silently skip missing ones.

```bash
sudo apt-get remove -y docker.io docker-doc docker-compose podman-docker containerd runc 2>&1 || true
```

Expected output: Either "Package X is not installed" lines or removal confirmations. No errors.

- [ ] **Step 2: Confirm removal**

```bash
dpkg -l | grep -E 'docker|containerd|runc' | grep '^ii' || echo "CLEAN — no conflicting packages installed"
```

Expected: `CLEAN — no conflicting packages installed`

- [ ] **Step 3: Commit checkpoint note to actions log**

```bash
cd /home/avik2007/mhw-risk-profiler
cat >> mhw_claude_actions/mhw_claude_recentactions.md << 'EOF'

## [2026-03-27] Docker Engine Install — Task 1 Complete

- Removed conflicting Ubuntu-repo Docker packages (docker.io, containerd, runc, etc.)
EOF
git add mhw_claude_actions/mhw_claude_recentactions.md
git commit -m "chore: docker install task 1 — remove conflicting packages"
```

---

### Task 2: Add Docker's official apt repository

**Files:**
- System files modified: `/etc/apt/keyrings/docker.asc`, `/etc/apt/sources.list.d/docker.list`

- [ ] **Step 1: Install apt prereqs**

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
```

Expected: Both packages already present or freshly installed. No errors.

- [ ] **Step 2: Create keyrings directory and download Docker's GPG key**

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

Expected: `/etc/apt/keyrings/docker.asc` exists, readable by all.

Verify:

```bash
ls -la /etc/apt/keyrings/docker.asc
```

Expected output contains `-rw-r--r--` permissions.

- [ ] **Step 3: Add Docker apt repository**

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

Expected: No output to terminal (piped to tee silently).

- [ ] **Step 4: Verify repo file contents**

```bash
cat /etc/apt/sources.list.d/docker.list
```

Expected output:
```
deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable
```

- [ ] **Step 5: Update apt index**

```bash
sudo apt-get update
```

Expected: Output includes `download.docker.com` in the fetched sources. No errors.

---

### Task 3: Install Docker Engine packages

**Files:**
- No project files modified — system package state only.

- [ ] **Step 1: Install all five Docker packages**

```bash
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
```

Expected: All five packages install without error. Output ends with "Processing triggers..." lines.

- [ ] **Step 2: Verify binaries are present**

```bash
which docker && docker --version
which docker && docker compose version
```

Expected:
```
/usr/bin/docker
Docker version 27.x.x, build xxxxxxx
Docker Compose version v2.x.x
```
(Exact versions will vary — any version line is sufficient.)

---

### Task 4: Configure docker group and daemon autostart

**Files:**
- No project files modified — system config only.

- [ ] **Step 1: Add user to docker group**

```bash
sudo usermod -aG docker avik2007
```

Expected: No output (silent success).

- [ ] **Step 2: Enable and start Docker daemon via systemd**

systemd is confirmed active (`ps -p 1 -o comm=` returns `systemd`).

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

Expected: No errors. `enable` may print a symlink creation line.

- [ ] **Step 3: Verify daemon is running**

```bash
sudo systemctl status docker --no-pager
```

Expected: Output contains `Active: active (running)`.

- [ ] **Step 4: Apply group membership without logging out**

```bash
newgrp docker
```

This opens a new shell with the `docker` group active. All subsequent commands in this session will work without `sudo`.

Note: On next login the group will apply automatically — `newgrp` is only needed for the current session.

---

### Task 5: Verification gate — all three checks must pass

**Files:**
- No project files modified.

- [ ] **Step 1: Run hello-world container**

```bash
docker run hello-world
```

Expected output contains:
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

If this fails with "permission denied", the `docker` group has not taken effect — run `newgrp docker` first (Task 4, Step 4).

- [ ] **Step 2: Confirm compose plugin**

```bash
docker compose version
```

Expected: `Docker Compose version v2.x.x` (any v2 version).

- [ ] **Step 3: Build the mhw-risk-profiler image**

```bash
cd /home/avik2007/mhw-risk-profiler
docker build -t mhw-risk-profiler .
```

Expected: Build completes. Final line is `Successfully built <hash>` or `=> => naming to docker.io/library/mhw-risk-profiler`. No errors.

This step exercises the full `Dockerfile`: apt system lib installs (`libgdal-dev`, `libnetcdf-dev`), `pip install -r requirements.txt`, and `COPY src/`. A successful build confirms the environment is production-ready.

- [ ] **Step 4: Update todo and actions logs**

```bash
cd /home/avik2007/mhw-risk-profiler
```

Update `mhw_claude_actions/mhw_claude_todo.md`: add Docker install to the COMPLETED section with date `[2026-03-27]`.

Update `mhw_claude_actions/mhw_claude_recentactions.md`: add a new entry:

```
## [2026-03-27] Docker Engine Installed and Verified

1. Removed conflicting Ubuntu-repo Docker packages.
2. Added Docker's official apt repo (Noble / amd64).
3. Installed: docker-ce, docker-ce-cli, containerd.io, docker-buildx-plugin, docker-compose-plugin.
4. Added avik2007 to docker group; enabled daemon via systemd.
5. Verification gate passed: hello-world OK, compose version OK, mhw-risk-profiler image built OK.
```

- [ ] **Step 5: Commit**

```bash
git add mhw_claude_actions/mhw_claude_todo.md mhw_claude_actions/mhw_claude_recentactions.md
git commit -m "chore: docker install complete — all verification checks passed"
```
