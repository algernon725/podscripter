### Install Docker on Windows, macOS, Ubuntu, and Bazzite (beginner-friendly)

This guide helps non-developers install Docker safely with links to the official documentation. If anything here differs from the official docs, follow the official docs.

On Bazzite (and other Fedora Atomic distributions) there is nothing to install: Podman ships with the system and works as a drop-in replacement for Docker. See the [Bazzite section](#bazzite-and-other-fedora-atomic-distros) below.

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

On Bazzite, substitute `podman` for `docker` in these (and all later) commands.

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

### Bazzite (and other Fedora Atomic distros)

**Use Podman on Bazzite — not Docker.**

Bazzite is an image-based ("immutable") Fedora Atomic system: the OS ships as a prebuilt image rather than a set of packages you install into. Podman is included in that image and is the container tool Bazzite expects you to use. It is daemonless, runs rootless by default, and accepts the same commands as Docker — anywhere the podscripter README says `docker`, type `podman`.

- Official docs:
  - Bazzite containers: `https://docs.bazzite.gg/Installing_and_Managing_Software/Containers/`
  - Podman: `https://docs.podman.io/en/latest/`

#### 1. Verify Podman (nothing to install)

```bash
podman --version
podman run hello-world
```

#### 2. Build and run podscripter

From the cloned `podscripter` folder, after creating the folders described in the README ("Set Up Required Folders"):

```bash
podman build -t podscripter .
```

```bash
podman run -it \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers:z \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface:z \
  -v $(pwd)/audio-files:/app/audio-files:z \
  podscripter
```

These are the README's commands with two differences: `podman` instead of `docker`, and a `:z` on the end of every `-v` mount.

#### 3. Why the `:z` matters

Bazzite runs with SELinux enforcing. Without a label, SELinux blocks the container from reading or writing your mounted folders, and you get `Permission denied` on the model caches or `audio-files/` even though the folders exist and look writable.

The `:z` suffix tells Podman to relabel that folder so containers may use it. Use lowercase `:z` (shared) rather than uppercase `:Z` (exclusive to one container) — these folders are reused across every run, and `:Z` would lock them to a single container.

#### 4. File ownership

Rootless Podman maps the container's root user to your own user account, so transcripts written to `audio-files/` are owned by you. No `sudo` and no ownership fixups are needed afterwards.

#### 5. The `docker-run-with-cache.sh` helper

That script calls `docker` directly, so it will not run on a stock Bazzite system. Either use the `podman run` command above, or install the Docker-compatible shim:

```bash
rpm-ostree install podman-docker   # reboot afterwards
```

For a shim that needs no reboot, add `alias docker=podman` to your `~/.bashrc`.

#### If you'd rather use Docker

Bazzite provides a helper recipe:

```bash
ujust install-docker   # reboot afterwards
```

This layers Docker onto the system image. Some users have reported the Docker service failing to start after this (`docker.service not found`), so Podman remains the smoother path for podscripter. Check the Bazzite docs above if you hit trouble.

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

- Bazzite/Podman:
  - `docker: command not found` is expected on a stock Bazzite system — use `podman`, or install the shim (`rpm-ostree install podman-docker`).
  - `Permission denied` when the container reads or writes a mounted folder almost always means a missing `:z` on that `-v` mount. Add it and re-run.
  - Anything installed with `rpm-ostree` (or `ujust`) only takes effect after a reboot.

---

### Uninstall/cleanup (optional)

Use the official docs for clean uninstallation and data removal:

- Windows/macOS (Docker Desktop): `https://docs.docker.com/desktop/`
- Ubuntu (Docker Desktop): `https://docs.docker.com/desktop/install/ubuntu/#uninstall-docker-desktop`
- Ubuntu (Docker Engine): `https://docs.docker.com/engine/install/ubuntu/#uninstall-docker-engine`

On Bazzite, Podman is part of the base system image and should not be removed. To clean up just this project, delete its image instead:

```bash
podman rmi podscripter
```


