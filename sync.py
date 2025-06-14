from __future__ import annotations

import copy
import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import jupytext


def get_jupytext_version() -> Optional[str]:
    """Checks for jupytext and returns its version."""
    try:
        result = subprocess.run(
            ["jupytext", "--version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sync_files(
    source_dir: str,
    dest_dir: str,
    source_ext: str,
    dest_ext: str,
    to_format: str,
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

                # Perform a robust comparison
                if dest_ext == ".ipynb":
                    # For notebooks (JSON), compare the parsed Python objects
                    # but ignore cell IDs which are randomly generated
                    try:
                        generated_json = json.loads(generated_dest_content)
                        on_disk_json = json.loads(on_disk_dest_content)

                        # Normalize notebooks by removing volatile fields such as
                        # cell IDs and kernelspec information which can vary
                        # between environments but do not affect notebook content.
                        def normalize_notebook(notebook_json: dict) -> dict:
                            nb_copy = copy.deepcopy(notebook_json)

                            # Strip random cell IDs
                            if "cells" in nb_copy:
                                for cell in nb_copy["cells"]:
                                    cell.pop("id", None)

                            # Remove kernelspec metadata that may differ across machines
                            if (
                                "metadata" in nb_copy
                                and "kernelspec" in nb_copy["metadata"]
                            ):
                                nb_copy["metadata"].pop("kernelspec", None)

                            return nb_copy

                        generated_clean = normalize_notebook(generated_json)
                        on_disk_clean = normalize_notebook(on_disk_json)

                        are_synced = generated_clean == on_disk_clean
                    except (json.JSONDecodeError, KeyError):
                        # Fallback to original comparison if JSON manipulation fails
                        are_synced = json.loads(generated_dest_content) == json.loads(
                            on_disk_dest_content
                        )
                else:
                    # For Markdown, a direct string comparison is sufficient
                    are_synced = generated_dest_content == on_disk_dest_content

            except Exception:
                # If any error occurs during parsing or comparison, assume not synced
                are_synced = False

            if are_synced:
                print(f"✅ 已同步: {source_file.name}")
            else:
                all_in_sync = False
                print(f"❌ 不同步: {source_file.name} -> {dest_file.name}")

                # Show the differences
                print("   📋 差異內容:")
                if dest_ext == ".ipynb":
                    # For notebooks, show formatted JSON diff
                    try:
                        generated_json = json.loads(generated_dest_content)
                        on_disk_json = json.loads(on_disk_dest_content)

                        # Pretty print normalized JSON for better readability
                        generated_formatted = json.dumps(
                            normalize_notebook(generated_json),
                            indent=2,
                            ensure_ascii=False,
                        )
                        on_disk_formatted = json.dumps(
                            normalize_notebook(on_disk_json),
                            indent=2,
                            ensure_ascii=False,
                        )

                        diff = difflib.unified_diff(
                            on_disk_formatted.splitlines(keepends=True),
                            generated_formatted.splitlines(keepends=True),
                            fromfile=f"{dest_file.name} (目前檔案)",
                            tofile=f"{dest_file.name} (來源轉換後)",
                            lineterm="",
                        )
                    except json.JSONDecodeError:
                        # Fallback to text diff if JSON parsing fails
                        diff = difflib.unified_diff(
                            on_disk_dest_content.splitlines(keepends=True),
                            generated_dest_content.splitlines(keepends=True),
                            fromfile=f"{dest_file.name} (目前檔案)",
                            tofile=f"{dest_file.name} (來源轉換後)",
                            lineterm="",
                        )
                else:
                    # For Markdown, show text diff
                    diff = difflib.unified_diff(
                        on_disk_dest_content.splitlines(keepends=True),
                        generated_dest_content.splitlines(keepends=True),
                        fromfile=f"{dest_file.name} (目前檔案)",
                        tofile=f"{dest_file.name} (來源轉換後)",
                        lineterm="",
                    )

                # Print the diff with proper indentation
                diff_lines = list(diff)
                if diff_lines:
                    for line in diff_lines[
                        :50
                    ]:  # Limit to first 50 lines to avoid overwhelming output
                        print(f"   {line.rstrip()}")
                    if len(diff_lines) > 50:
                        print(f"   ... (還有 {len(diff_lines) - 50} 行差異)")
                else:
                    print("   (無法顯示差異)")

                if dest_file.stat().st_mtime > source_file.stat().st_mtime:
                    print(
                        f"   ⚠️  警告: 目標檔案 {dest_file.name} 較新，覆寫將會遺失其變更。"
                    )

                choice = input(
                    f"   您要用 {source_ext} 的內容覆寫 {dest_ext} 嗎？ (y/N): "
                ).lower()
                if choice == "y":
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
        sync_files("notebook_md", "notebook", ".md", ".ipynb", "ipynb")
    elif direction == "nb2md":
        print("=== Interactive Sync: Notebooks to Markdown (.ipynb -> .md) ===")
        sync_files("notebook", "notebook_md", ".ipynb", ".md", "md")


if __name__ == "__main__":
    main()
