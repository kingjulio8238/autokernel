"""Deploy the openkernel Modal eval app.

Usage:
    # Deploy to Modal (creates/updates the app)
    modal deploy modal_infra/app.py

    # Or use this script for deployment with pre-flight checks:
    python modal_infra/deploy.py
    python modal_infra/deploy.py --check    # health check only
    python modal_infra/deploy.py --dry-run  # validate without deploying
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def check_modal_auth() -> bool:
    """Verify Modal CLI is authenticated."""
    result = subprocess.run(
        ["modal", "token", "show"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def deploy_app(dry_run: bool = False) -> int:
    """Deploy the Modal app."""
    cmd = ["modal", "deploy", "modal_infra/app.py"]
    if dry_run:
        print("[DRY RUN] Would execute:", " ".join(cmd))
        # Validate the app can be parsed by importing it
        try:
            import modal_infra.app  # noqa: F401

            print("[OK] App module imports successfully.")
            return 0
        except Exception as exc:
            print(f"[ERROR] App module failed to import: {exc}")
            return 1

    print(f"Deploying: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_health_check() -> int:
    """Call the health_check function on the deployed app."""
    try:
        from modal_infra.app import health_check

        print("Running health check on deployed app...")
        result = health_check.remote()
        print(f"Health check result: {result}")
        if result.get("status") == "ok" and result.get("cuda_available"):
            print("[OK] GPU container is healthy.")
            return 0
        else:
            print("[WARN] Container responded but GPU may not be available.")
            return 1
    except Exception as exc:
        print(f"[ERROR] Health check failed: {exc}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy openkernel Modal eval app")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run health check on the deployed app (skip deployment)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the app without deploying",
    )
    args = parser.parse_args()

    # Pre-flight: check Modal auth
    if not check_modal_auth():
        print("[ERROR] Modal is not authenticated. Run: modal token new")
        sys.exit(1)
    print("[OK] Modal authentication verified.")

    if args.check:
        sys.exit(run_health_check())

    # Deploy
    rc = deploy_app(dry_run=args.dry_run)
    if rc != 0:
        print("[ERROR] Deployment failed.")
        sys.exit(rc)

    if not args.dry_run:
        print("[OK] Deployment successful.")
        # Run health check after deploy
        print("\nRunning post-deploy health check...")
        run_health_check()


if __name__ == "__main__":
    main()
