#!/usr/bin/env python3
"""
Batch DICOM → NIfTI converter using dcm2niix.
Copies each ADNI series to a fast scratch SSD, converts it with dcm2niix, 
then moves the .nii/.nii.gz back to the output tree.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _is_dicom_file(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() in {"", ".dcm", ".dicom"})


def _series_dirs(root: Path) -> list[Path]:

    """Return every directory that contains at least one DICOM slice."""
    out: list[Path] = []
    for d in root.rglob("*"):
        if d.is_dir():
            try:
                next(f for f in d.iterdir() if _is_dicom_file(f))
                out.append(d)
            except StopIteration:
                continue
    return out


def _check_dcm2niix() -> bool:
    """Check if dcm2niix is available in PATH."""
    try:
        result = subprocess.run(
            ["dcm2niix", "-h"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _convert_one_series(
    series_dir: Path,
    input_root: Path,
    output_root: Path,
    scratch_root: Path,
    *,
    compress: bool,
    force_conversion: bool = False,
) -> bool:
    """Convert one DICOM series to NIfTI using dcm2niix."""
    
    rel = series_dir.relative_to(input_root)
    dest_dir = output_root / rel.parent  # place NIfTI one level up
    dest_dir.mkdir(parents=True, exist_ok=True)

    if any(dest_dir.glob("*.nii*")):
        logger.info("Skipping %s (already converted)", rel)
        return True

    with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
        tmp_dicom = Path(tmp) / "dicom"
        tmp_output = Path(tmp) / "output"
        tmp_output.mkdir()
        
        # Copy DICOM files to scratch
        shutil.copytree(series_dir, tmp_dicom)
        
        # Build dcm2niix command
        cmd = [
            "dcm2niix",
            "-o", str(tmp_output),  # output directory
            "-f", "%p_%s",          # filename format: protocol_seriesnum
            "-z", "y" if compress else "n",  # compress output
            "-b", "y",              # create BIDS sidecar JSON
            "-v", "1",              # verbose level
        ]
        
        # Add force options if requested
        if force_conversion:
            cmd.extend([
                "-i", "y",          # ignore derived, localizer, 2D images
                "-m", "y",          # merge 2D slices from same series
                "-s", "y",          # single file mode
            ])
        
        cmd.append(str(tmp_dicom))
        
        try:
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=tmp
            )
            
            if result.returncode == 0:
                # Move converted files to destination
                converted_files = list(tmp_output.glob("*.nii*"))
                json_files = list(tmp_output.glob("*.json"))
                
                if converted_files:
                    for nii_file in converted_files:
                        shutil.move(str(nii_file), dest_dir / nii_file.name)
                    
                    # Also move JSON sidecar files
                    for json_file in json_files:
                        shutil.move(str(json_file), dest_dir / json_file.name)
                    
                    logger.info("✓ Converted %s → %s (%d files)", rel, dest_dir, len(converted_files))
                    return True
                else:
                    logger.warning("❌ No NIfTI files produced for %s", rel)
                    _log_conversion_error(dest_dir, rel, result.stdout, result.stderr)
                    return False
            else:
                logger.warning("❌ dcm2niix failed for %s (exit code: %d)", rel, result.returncode)
                _log_conversion_error(dest_dir, rel, result.stdout, result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("❌ Conversion timeout for %s", rel)
            return False
        except Exception as exc:
            logger.warning("❌ Conversion error for %s: %s", rel, exc)
            return False


def _log_conversion_error(dest_dir: Path, rel: Path, stdout: str, stderr: str) -> None:
    """Log detailed error information to a file."""
    error_log = dest_dir / f"{rel.name}_error.log"
    with open(error_log, 'w') as f:
        f.write(f"Conversion failed for: {rel}\n")
        f.write(f"Timestamp: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n")
        f.write("\n--- STDOUT ---\n")
        f.write(stdout)
        f.write("\n--- STDERR ---\n")
        f.write(stderr)
    logger.info("Error details saved to: %s", error_log)


def convert_dicom_tree_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    scratch_dir: Union[str, Path] | None = None,
    compress: bool = True,
    force_conversion: bool = False,
) -> None:
    """Convert a tree of DICOM series to NIfTI using dcm2niix."""
    
    # Check if dcm2niix is available
    if not _check_dcm2niix():
        logger.error("dcm2niix not found in PATH. Please install it:")
        logger.error("  macOS: brew install dcm2niix")
        logger.error("  conda: conda install -c conda-forge dcm2niix")
        logger.error("  or download from: https://github.com/rordenlab/dcm2niix")
        return

    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    scratch_root = (
        Path(scratch_dir).expanduser().resolve()
        if scratch_dir else Path(tempfile.gettempdir())
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)

    logger.info("Input   : %s", input_dir)
    logger.info("Output  : %s", output_dir)
    logger.info("Scratch : %s", scratch_root)
    logger.info("Compress: %s", "ON (.nii.gz)" if compress else "OFF (.nii)")
    logger.info("Force conversion: %s", "ON" if force_conversion else "OFF")

    series = _series_dirs(input_dir)
    logger.info("Found %d series", len(series))

    successful_conversions = 0
    failed_conversions = 0

    for idx, sdir in enumerate(series, 1):
        logger.info("[%d/%d] %s", idx, len(series), sdir.relative_to(input_dir))
        
        try:
            success = _convert_one_series(
                sdir,
                input_root=input_dir,
                output_root=output_dir,
                scratch_root=scratch_root,
                compress=compress,
                force_conversion=force_conversion,
            )
            
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
                
        except Exception as e:
            logger.error("Unexpected error processing %s: %s", sdir.relative_to(input_dir), e)
            failed_conversions += 1

    logger.info("All done.")
    logger.info("Summary: %d successful, %d failed conversions", successful_conversions, failed_conversions)
    
    if failed_conversions > 0:
        logger.info("Check error log files in output directories for details on failed conversions")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch DICOM → NIfTI using dcm2niix")
    p.add_argument("input_dir", type=Path, help="Root directory of DICOM tree")
    p.add_argument("output_dir", type=Path, help="Destination for NIfTI files")
    p.add_argument("--scratch-dir", type=Path, default=None,
                   help="Local SSD dir for temp work (default: system /tmp)")
    p.add_argument("--no-compress", action="store_true",
                   help="Write .nii instead of .nii.gz")
    p.add_argument("--force", action="store_true",
                   help="Force conversion with additional dcm2niix options")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    convert_dicom_tree_batch(
        args.input_dir,
        args.output_dir,
        scratch_dir=args.scratch_dir,
        compress=not args.no_compress,
        force_conversion=args.force,
    )


if __name__ == "__main__":
    main()
