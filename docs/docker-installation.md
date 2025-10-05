### Install Docker on Windows, macOS, and Ubuntu (beginner-friendly)

This guide helps non-developers install Docker safely with links to the official documentation. If anything here differs from the official docs, follow the official docs.

---

### What is Docker?

Docker lets you run apps in containers (lightweight, isolated environments). You don't need to be a developer to install it.

---

### Quick verification (works on all platforms)

After installing Docker, open your terminal (PowerShell on Windows, Terminal on macOS/Ubuntu) and run:

```bash
docker --version
docker run hello-world
```

You should see a success message from the `hello-world` container.

---

### Windows (recommended: Docker Desktop)

Docker Desktop for Windows is the easiest option.

- Requirements: Windows 10/11 64-bit, virtualization enabled, and WSL 2.
- Official docs:
  - Docker Desktop for Windows: `https://docs.docker.com/desktop/install/windows-install/`
  - WSL 2 (Windows Subsystem for Linux): `https://learn.microsoft.com/windows/wsl/install`

Steps:
1. Install WSL 2 using the Microsoft guide above (a restart may be required).
2. Download Docker Desktop for Windows from the Docker docs page.
3. Run the installer. When prompted, keep the WSL 2 backend selected.
4. Start Docker Desktop (allow any Windows Firewall prompts).
5. Open PowerShell and verify:
   - `docker --version`
   - `docker run hello-world`

---

### macOS (recommended: Docker Desktop)

Docker Desktop for Mac supports both Apple Silicon (M-series) and Intel Macs.

- Requirements: macOS 11+ recommended, enough disk space and memory to run containers.
- Official docs:
  - Docker Desktop for Mac: `https://docs.docker.com/desktop/install/mac-install/`

Steps:
1. Download Docker Desktop for Mac from the Docker docs page.
2. Open the `.dmg` and drag Docker to `Applications`.
3. Open Docker from `Applications` and grant any requested permissions.
4. Open Terminal and verify:
   - `docker --version`
   - `docker run hello-world`

---

### Ubuntu Linux

You have two common choices:

1) Docker Desktop for Linux (simpler UI, similar to Windows/macOS):
   - Official docs: `https://docs.docker.com/desktop/install/ubuntu/`
   - Summary: download the `.deb` for your Ubuntu version from the Docker page, install it (e.g., `sudo apt install ./docker-desktop-<version>-<arch>.deb`), then launch Docker Desktop.

2) Docker Engine (command-line; lightweight; very common on servers):
   - Official docs: `https://docs.docker.com/engine/install/ubuntu/`
   - Post-install (use Docker without `sudo`): `https://docs.docker.com/engine/install/linux-postinstall/`

Verification (either option):
```bash
docker --version
docker run hello-world
```

If you used Docker Engine and want to run Docker without `sudo`, follow the Linux post-install guide to add your user to the `docker` group, then log out and back in.

---

### Common troubleshooting

- Network/firewall/proxy:
  - Docker and Docker Desktop need outbound internet access on HTTPS (port 443).
  - On Ubuntu with UFW, check: `sudo ufw status verbose`. If outbound is blocked, consider `sudo ufw default allow outgoing` or allow specific ports: `sudo ufw allow out 443/tcp`.
  - If you use a corporate proxy, configure proxy settings in Docker Desktop (Settings) or set environment variables (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`).

- Windows WSL 2 integration:
  - In Docker Desktop Settings, ensure WSL 2 integration is enabled for your WSL distributions.

- Permissions on Linux:
  - If `docker run hello-world` fails with a permissions error, follow the Linux post-install guide to use Docker as a non-root user.

- DNS/connectivity test:
  - Try `curl https://docs.docker.com` to confirm you can reach Docker docs from your machine.

---

### Uninstall/cleanup (optional)

Use the official docs for clean uninstallation and data removal:

- Windows/macOS (Docker Desktop): `https://docs.docker.com/desktop/`
- Ubuntu (Docker Desktop): `https://docs.docker.com/desktop/install/ubuntu/#uninstall-docker-desktop`
- Ubuntu (Docker Engine): `https://docs.docker.com/engine/install/ubuntu/#uninstall-docker-engine`


