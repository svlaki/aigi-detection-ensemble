"""Copy cs231n-data volume from cs-231n workspace to svlaki workspace.

Runs a function in each workspace:
  1. cs-231n: reads all files from the volume, returns them as bytes
  2. svlaki: writes those bytes into a new volume

Usage:
  # Step 1: Export from cs-231n (run with cs-231n profile active)
  modal profile activate cs-231n
  modal run scripts/transfer_volume.py --cmd export

  # Step 2: Import to svlaki (run with svlaki profile active)
  modal profile activate svlaki
  modal run scripts/transfer_volume.py --cmd import
"""
import modal
import os
import sys

app = modal.App("volume-transfer")
DUMP_DIR = "/tmp/modal-volume-dump"


@app.local_entrypoint()
def main(cmd: str = "export"):
    if cmd == "export":
        export_volume()
    elif cmd == "import":
        import_volume()
    else:
        print(f"Unknown command: {cmd}. Use 'export' or 'import'.")
        sys.exit(1)


def export_volume():
    """Download every file from the volume to a local staging directory."""
    import shutil
    from pathlib import Path

    vol = modal.Volume.from_name("cs231n-data")

    staging = Path(DUMP_DIR)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    count = 0
    for entry in vol.listdir("/", recursive=True):
        if entry.type.name == "FILE":
            rpath = entry.path.lstrip("/")
            local_path = staging / rpath
            local_path.parent.mkdir(parents=True, exist_ok=True)

            data = b""
            for chunk in vol.read_file(entry.path):
                data += chunk
            local_path.write_bytes(data)
            count += 1
            if count % 500 == 0:
                print(f"  exported {count} files...")

    print(f"Exported {count} files to {staging}")


def import_volume():
    """Upload the local staging directory into a new volume."""
    from pathlib import Path

    staging = Path(DUMP_DIR)
    if not staging.exists():
        print(f"ERROR: {staging} not found. Run 'export' first.")
        sys.exit(1)

    vol = modal.Volume.from_name("cs231n-data", create_if_missing=True)
    print("Using volume cs231n-data (created if missing).")

    all_files = sorted(f for f in staging.rglob("*") if f.is_file())
    print(f"Uploading {len(all_files)} files...")

    with vol.batch_upload() as batch:
        for f in all_files:
            remote_path = "/" + str(f.relative_to(staging))
            batch.put_file(f, remote_path)

    print(f"Uploaded {len(all_files)} files to cs231n-data volume.")
