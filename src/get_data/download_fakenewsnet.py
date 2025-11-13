#!/usr/bin/env python3
import subprocess, time, shutil, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "FakeNewsNet" / time.strftime("%Y-%m-%d")
REPO_URL = "https://github.com/KaiDMML/FakeNewsNet.git"

def run(cmd): subprocess.check_call(cmd, shell=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # make a temp folder for the clone
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"[info] Cloning FakeNewsNet temporarily → {tmp_path}")
        run(f'git clone --depth=1 "{REPO_URL}" "{tmp_path}"')

        # look for /dataset or /code/dataset
        for d in [tmp_path / "dataset", tmp_path / "code" / "dataset"]:
            if d.is_dir():
                for csv in d.glob("*.csv"):
                    shutil.copy2(csv, OUT / csv.name)
                print(f"[done] Copied {len(list(d.glob('*.csv')))} CSVs from {d}")
                break
        else:
            print("[warn] No dataset folder found!")

    print(f"Temporary clone deleted automatically.")
    print(f"Files saved under: {OUT}")

if __name__ == "__main__":
    main()
