"""Create/update the public Hugging Face Docker Space for the demo."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "deploy" / "huggingface"
REPO_ID = "ldidone/reingresos-diabetes-demo"

FILES = {
    "README.md": CONFIG / "README.md",
    "Dockerfile": CONFIG / "Dockerfile",
    "requirements.txt": CONFIG / "requirements.txt",
    ".streamlit/config.toml": ROOT / ".streamlit" / "config.toml",
    "reingresos_utils.py": ROOT / "reingresos_utils.py",
    "demo/app.py": ROOT / "demo" / "app.py",
    "demo/demo_logic.py": ROOT / "demo" / "demo_logic.py",
    "demo/artifacts/readmission_demo.joblib": (
        ROOT / "demo" / "artifacts" / "readmission_demo.joblib"
    ),
}


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing deployment files: {missing}")

    api = HfApi()
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    api.update_repo_settings(REPO_ID, repo_type="space", private=False)
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
            for remote, local in FILES.items()
        ],
        commit_message="Deploy Streamlit readmission demo",
        commit_description=(
            "Frozen LightGBM v2 model with Platt calibration, interactive patient "
            "profiles, local sensitivity explanations, and operational impact."
        ),
    )
    info = api.repo_info(REPO_ID, repo_type="space")
    if info.private:
        raise RuntimeError("The Space was not made public.")
    print(f"Commit: {commit.commit_url}")
    print(f"Space: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
