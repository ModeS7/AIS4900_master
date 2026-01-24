#!/usr/bin/env python3
"""
Pack multiple folders into a single archive file or extract them.

Usage:
  # Pack folders into archive
  python misc/archive_folders.py pack --folders /path/to/folder1 /path/to/folder2 --output my_archive.tar.gz

  # Extract folders from archive
  python misc/archive_folders.py unpack --archive my_archive.tar.gz --output-dir /path/to/extract

  # Dry run (see what would be packed)
  python misc/archive_folders.py pack --folders /path/to/folder1 --output test.tar.gz --dry-run

Supported formats: .tar.gz, .tar.bz2, .tar.xz, .zip
"""

import argparse
import tarfile
import zipfile
from pathlib import Path
from typing import List
import sys


def get_dir_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def pack_folders_tar(folders: List[Path], output_file: Path, compression: str, dry_run: bool = False):
    """Pack folders into tar archive."""

    # Determine compression mode
    if compression == 'gz':
        mode = 'w:gz'
    elif compression == 'bz2':
        mode = 'w:bz2'
    elif compression == 'xz':
        mode = 'w:xz'
    else:
        mode = 'w'

    print(f"\nPacking {len(folders)} folder(s) into {output_file}")
    print(f"Compression: {compression if compression else 'none'}")
    print("="*60)

    # Calculate total size
    total_size = 0
    for folder in folders:
        size = get_dir_size(folder)
        total_size += size
        print(f"  {folder.name}: {format_size(size)}")

    print(f"\nTotal uncompressed size: {format_size(total_size)}")

    if dry_run:
        print("\n[DRY RUN] No archive was created.")
        return

    # Create archive
    print("\nCreating archive...")
    with tarfile.open(output_file, mode) as tar:
        for folder in folders:
            print(f"  Adding {folder.name}...")
            tar.add(folder, arcname=folder.name)

    # Show final size
    archive_size = output_file.stat().st_size
    compression_ratio = (1 - archive_size / total_size) * 100 if total_size > 0 else 0

    print(f"\n[SUCCESS] Archive created: {output_file}")
    print(f"Archive size: {format_size(archive_size)}")
    print(f"Compression ratio: {compression_ratio:.1f}%")


def pack_folders_zip(folders: List[Path], output_file: Path, dry_run: bool = False):
    """Pack folders into zip archive."""

    print(f"\nPacking {len(folders)} folder(s) into {output_file}")
    print("Compression: zip")
    print("="*60)

    # Calculate total size
    total_size = 0
    for folder in folders:
        size = get_dir_size(folder)
        total_size += size
        print(f"  {folder.name}: {format_size(size)}")

    print(f"\nTotal uncompressed size: {format_size(total_size)}")

    if dry_run:
        print("\n[DRY RUN] No archive was created.")
        return

    # Create archive
    print("\nCreating archive...")
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in folders:
            print(f"  Adding {folder.name}...")
            for file_path in folder.rglob('*'):
                if file_path.is_file():
                    arcname = str(file_path.relative_to(folder.parent))
                    zf.write(file_path, arcname=arcname)

    # Show final size
    archive_size = output_file.stat().st_size
    compression_ratio = (1 - archive_size / total_size) * 100 if total_size > 0 else 0

    print(f"\n[SUCCESS] Archive created: {output_file}")
    print(f"Archive size: {format_size(archive_size)}")
    print(f"Compression ratio: {compression_ratio:.1f}%")


def unpack_tar(archive_file: Path, output_dir: Path, dry_run: bool = False):
    """Extract tar archive."""

    print(f"\nExtracting {archive_file} to {output_dir}")
    print("="*60)

    with tarfile.open(archive_file, 'r:*') as tar:
        members = tar.getmembers()

        # Show contents
        folders = set()
        total_size = 0
        for member in members:
            if member.isdir():
                folders.add(member.name.split('/')[0])
            total_size += member.size

        print(f"Archive contains {len(folders)} top-level folder(s):")
        for folder in sorted(folders):
            print(f"  - {folder}")
        print(f"\nTotal size: {format_size(total_size)}")

        if dry_run:
            print("\n[DRY RUN] No files were extracted.")
            return

        # Extract
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExtracting {len(members)} items...")
        tar.extractall(output_dir)

    print(f"\n[SUCCESS] Extracted to {output_dir}")


def unpack_zip(archive_file: Path, output_dir: Path, dry_run: bool = False):
    """Extract zip archive."""

    print(f"\nExtracting {archive_file} to {output_dir}")
    print("="*60)

    with zipfile.ZipFile(archive_file, 'r') as zf:
        members = zf.namelist()

        # Show contents
        folders = set()
        total_size = 0
        for member in members:
            folders.add(member.split('/')[0])
            total_size += zf.getinfo(member).file_size

        print(f"Archive contains {len(folders)} top-level folder(s):")
        for folder in sorted(folders):
            print(f"  - {folder}")
        print(f"\nTotal size: {format_size(total_size)}")

        if dry_run:
            print("\n[DRY RUN] No files were extracted.")
            return

        # Extract
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExtracting {len(members)} items...")
        zf.extractall(output_dir)

    print(f"\n[SUCCESS] Extracted to {output_dir}")


def cmd_pack(args):
    """Pack folders into archive."""

    # Validate folders
    folders = []
    for folder_path in args.folders:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"[ERROR] Folder does not exist: {folder}")
            return 1
        if not folder.is_dir():
            print(f"[ERROR] Not a directory: {folder}")
            return 1
        folders.append(folder)

    output_file = Path(args.output)

    # Check if output exists
    if output_file.exists() and not args.dry_run:
        response = input(f"\n[WARNING] Output file exists: {output_file}\nOverwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return 0

    # Determine format from extension
    suffix = output_file.suffix.lower()

    try:
        if suffix == '.zip':
            pack_folders_zip(folders, output_file, args.dry_run)
        elif suffix == '.gz':
            pack_folders_tar(folders, output_file, 'gz', args.dry_run)
        elif suffix == '.bz2':
            pack_folders_tar(folders, output_file, 'bz2', args.dry_run)
        elif suffix == '.xz':
            pack_folders_tar(folders, output_file, 'xz', args.dry_run)
        elif output_file.suffixes[-2:] == ['.tar', '.gz']:
            pack_folders_tar(folders, output_file, 'gz', args.dry_run)
        elif output_file.suffixes[-2:] == ['.tar', '.bz2']:
            pack_folders_tar(folders, output_file, 'bz2', args.dry_run)
        elif output_file.suffixes[-2:] == ['.tar', '.xz']:
            pack_folders_tar(folders, output_file, 'xz', args.dry_run)
        else:
            pack_folders_tar(folders, output_file, None, args.dry_run)
    except Exception as e:
        print(f"\n[ERROR] Failed to create archive: {e}")
        return 1

    return 0


def cmd_unpack(args):
    """Extract archive."""

    archive_file = Path(args.archive)

    # Validate archive
    if not archive_file.exists():
        print(f"[ERROR] Archive does not exist: {archive_file}")
        return 1

    if not archive_file.is_file():
        print(f"[ERROR] Not a file: {archive_file}")
        return 1

    output_dir = Path(args.output_dir)

    # Check if output directory exists
    if output_dir.exists() and not args.dry_run:
        response = input(f"\n[WARNING] Output directory exists: {output_dir}\nExtract anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return 0

    # Determine format
    suffix = archive_file.suffix.lower()

    try:
        if suffix == '.zip':
            unpack_zip(archive_file, output_dir, args.dry_run)
        elif suffix in ['.gz', '.bz2', '.xz'] or archive_file.suffixes[-2:][0] == '.tar':
            unpack_tar(archive_file, output_dir, args.dry_run)
        else:
            # Try tar first, then zip
            try:
                unpack_tar(archive_file, output_dir, args.dry_run)
            except:
                unpack_zip(archive_file, output_dir, args.dry_run)
    except Exception as e:
        print(f"\n[ERROR] Failed to extract archive: {e}")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Pack folders into archive or extract them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pack two folders into compressed archive
  python misc/archive_folders.py pack --folders /data/train /data/test --output dataset.tar.gz

  # Extract archive
  python misc/archive_folders.py unpack --archive dataset.tar.gz --output-dir /data/extracted

  # Dry run to see what would be packed
  python misc/archive_folders.py pack --folders /data/train --output test.tar.gz --dry-run

Supported formats:
  - .tar.gz (recommended - good compression, fast)
  - .tar.bz2 (better compression, slower)
  - .tar.xz (best compression, slowest)
  - .zip (Windows-compatible)
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Pack command
    pack_parser = subparsers.add_parser('pack', help='Pack folders into archive')
    pack_parser.add_argument('--folders', nargs='+', required=True,
                            help='Folders to pack')
    pack_parser.add_argument('--output', '-o', required=True,
                            help='Output archive file')
    pack_parser.add_argument('--dry-run', action='store_true',
                            help='Show what would be done without creating archive')

    # Unpack command
    unpack_parser = subparsers.add_parser('unpack', help='Extract archive')
    unpack_parser.add_argument('--archive', '-a', required=True,
                              help='Archive file to extract')
    unpack_parser.add_argument('--output-dir', '-o', required=True,
                              help='Directory to extract to')
    unpack_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be extracted without extracting')

    args = parser.parse_args()

    if args.command == 'pack':
        return cmd_pack(args)
    elif args.command == 'unpack':
        return cmd_unpack(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
