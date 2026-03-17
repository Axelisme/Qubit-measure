from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import jupytext  # pyright: ignore
from typing_extensions import Optional

# ------------------------------
# Helper utilities
# ------------------------------


def get_jupytext_version() -> Optional[str]:
    """Checks for jupytext and returns its version."""
    try:
        result = subprocess.run(
            ["jupytext", "--version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _normalize_notebook(nb: dict) -> dict:
    """Return a deep-copied notebook dict with unstable fields removed."""
    nb_copy = copy.deepcopy(nb)

    # Remove random cell IDs
    if "cells" in nb_copy:
        for cell in nb_copy["cells"]:
            cell.pop("id", None)

    # Remove environment-specific metadata
    if "metadata" in nb_copy:
        nb_copy["metadata"].pop("kernelspec", None)
        nb_copy["metadata"].pop("language_info", None)

    return nb_copy


# ------------------------------
# Main function
# ------------------------------


def sync_files(
    source_dir: str,
    dest_dir: str,
    source_ext: str,
    dest_ext: str,
    to_format: str,
    *,
    always_overwrite: bool = False,
    prompt: bool = True,
    default_yes: bool = False,
) -> None:
    """Synchronize ``source_ext`` files in *source_dir* to ``dest_ext`` files in *dest_dir*.

    Differences are displayed and the user is asked whether to overwrite when content
    is not in sync.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.is_dir():
        print(f"❌ 錯誤: 找不到來源目錄 '{source_path}'。")
        sys.exit(1)

    all_source_files = list(source_path.rglob(f"*{source_ext}"))
    if not all_source_files:
        print(f"ℹ️ 在 '{source_path}/' 目錄中找不到任何 {source_ext} 檔案。")
        return

    synced_count = 0
    processed_count = 0
    all_in_sync = True

    print("🔎 開始檢查...")
    for source_file in all_source_files:
        processed_count += 1
        relative_path = source_file.relative_to(source_path)
        dest_file = dest_path / relative_path.with_suffix(dest_ext)

        # Pre-convert source to destination format for comparison/writing
        source_notebook: Optional[dict] = None

        # -------------------------------------------------------------
        # Fast-path: unconditional overwrite requested (e.g. nb2md mode)
        # -------------------------------------------------------------
        if always_overwrite:
            if source_notebook is None:
                source_notebook = jupytext.read(source_file)

            if not dest_file.exists():
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                print(f"ℹ️  正在建立新檔案: {source_file.name} -> {dest_file.name}")
            else:
                print(f"🔄 覆寫: {source_file.name} -> {dest_file.name}")

            jupytext.write(source_notebook, dest_file)
            synced_count += 1
            continue  # move to next file

        # -------------------------------------------------------------
        # Regular interactive/compare workflow
        # -------------------------------------------------------------

        if not dest_file.exists():
            all_in_sync = False
            print(f"ℹ️  正在建立新檔案: {source_file.name} -> {dest_file.name}")
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "jupytext",
                    "--to",
                    to_format,
                    str(source_file),
                    "--output",
                    str(dest_file),
                ],
                check=True,
            )
            print("   ✅ 已建立。")
            synced_count += 1
        else:
            # Replace the CLI diff check with a more robust in-memory comparison
            # using the jupytext library directly.
            try:
                # Read the source file and convert it to the destination format in memory
                source_notebook = jupytext.read(source_file)
                generated_dest_content = jupytext.writes(
                    source_notebook, fmt=dest_ext.strip(".")
                )

                # Read the actual content of the destination file from disk
                on_disk_dest_content = dest_file.read_text(encoding="utf-8")

                # 使用全域 _normalize_notebook 進行比較
                if dest_ext == ".ipynb":
                    try:
                        generated_json = json.loads(generated_dest_content)
                        on_disk_json = json.loads(on_disk_dest_content)

                        generated_clean = _normalize_notebook(generated_json)
                        on_disk_clean = _normalize_notebook(on_disk_json)

                        are_synced = generated_clean == on_disk_clean
                    except (json.JSONDecodeError, KeyError):
                        are_synced = json.loads(generated_dest_content) == json.loads(
                            on_disk_dest_content
                        )
                else:
                    try:
                        generated_nb = jupytext.reads(
                            generated_dest_content, fmt=dest_ext.strip(".")
                        )
                        on_disk_nb = jupytext.read(dest_file)

                        generated_clean = _normalize_notebook(generated_nb)
                        on_disk_clean = _normalize_notebook(on_disk_nb)

                        are_synced = generated_clean == on_disk_clean
                    except Exception:
                        are_synced = generated_dest_content == on_disk_dest_content

            except Exception:
                # If any error occurs during parsing or comparison, assume not synced
                are_synced = False

            if are_synced:
                print(f"✅ 已同步: {source_file.name}")
            else:
                all_in_sync = False
                print(f"❌ 不同步: {source_file.name} -> {dest_file.name}")

                if dest_file.stat().st_mtime > source_file.stat().st_mtime:
                    print(
                        f"   ⚠️  警告: 目標檔案 {dest_file.name} 較新，覆寫將會遺失其變更。"
                    )

                if prompt:
                    default_prompt = "Y/n" if default_yes else "y/N"
                    choice = input(
                        f"   您要用 {source_ext} 的內容覆寫 {dest_ext} 嗎？ ({default_prompt}): "
                    ).lower()

                    overwrite = False
                    if default_yes:
                        overwrite = choice not in ["n", "no"]
                    else:
                        overwrite = choice in ["y", "yes"]
                else:
                    overwrite = True  # non-interactive path

                if overwrite:
                    # Use jupytext library to perform the conversion
                    jupytext.write(source_notebook, dest_file)
                    print(f"   🔄 已同步: {dest_file.name}")
                    synced_count += 1
                else:
                    print("   ⏩ 已跳過")

    print("\n---\n✨ 檢查完成！")
    if all_in_sync:
        print(f"✅ 所有 {processed_count} 個檔案都已同步。")
    else:
        print(f"總共處理 {processed_count} 個檔案，同步/建立了 {synced_count} 個檔案。")


def main() -> None:
    """Main function to handle command-line arguments."""
    if get_jupytext_version() is None:
        print("❌ 錯誤: 找不到 jupytext！請先啟用對應的 Conda/Python 環境。")
        sys.exit(1)

    if len(sys.argv) != 2 or sys.argv[1] not in ["md2nb", "nb2md"]:
        print("用法: python sync.py [md2nb|nb2md]")
        sys.exit(1)

    direction = sys.argv[1]

    if direction == "md2nb":
        print("=== Interactive Sync: Markdown to Notebooks (.md -> .ipynb) ===")
        sync_files(
            "notebook_md",
            "notebook",
            ".md",
            ".ipynb",
            "ipynb",
            prompt=True,
            default_yes=True,
        )
    elif direction == "nb2md":
        print("=== Interactive Sync: Notebooks to Markdown (.ipynb -> .md) ===")
        sync_files(
            "notebook",
            "notebook_md",
            ".ipynb",
            ".md",
            "md",
            always_overwrite=True,
            prompt=False,
        )


if __name__ == "__main__":
    main()
