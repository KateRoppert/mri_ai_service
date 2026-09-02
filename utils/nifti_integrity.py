"""
Is a NIfTI file complete, or was it truncated mid-write?

Pipeline stages decide "already processed, skip it" from the presence of an
output file. That test is wrong for any file whose writer died partway
through — a stopped run, a full disk, an OOM kill — because a truncated
.nii.gz still has a valid header and still satisfies exists(). nibabel.load()
does not help either: it reads the header and returns happily.

The end of the file is where the truth is. A gzip stream records the
uncompressed size of its payload in the last four bytes, and the NIfTI header
states how many bytes of image data there should be; if the two disagree, the
file is not whole.
"""

import logging
import struct
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# NIfTI-1 puts image data at byte 352 when vox_offset is left at 0.
_NIFTI1_HEADER_BYTES = 352

# gzip's ISIZE field is the payload size modulo 2**32, so this check cannot
# distinguish a 4 GiB file from an empty one. Our volumes are ~30 MB; refuse
# to guess above the limit rather than return a confident wrong answer.
_ISIZE_MODULUS = 2 ** 32

# Header extensions legitimately sit between the header and the data, so the
# trailer may exceed the computed minimum by a little. A real truncation is
# off by megabytes, never by kilobytes.
_EXTENSION_SLACK_BYTES = 65536


def is_complete_nifti(path: Union[Path, str]) -> bool:
    """
    True if `path` is a NIfTI file whose image data is fully present.

    Errs toward False: an unreadable or unrecognisable file is reported
    incomplete. Recomputing a good file costs minutes; trusting a bad one
    puts a wrong number in a clinical report.
    """
    path = Path(path)

    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
    except OSError as e:
        logger.debug("cannot stat %s: %s", path, e)
        return False

    try:
        img = nib.load(str(path))
        header = img.header
        data_bytes = int(np.prod(img.shape)) * header.get_data_dtype().itemsize
        # vox_offset is 0 in files written without extensions; the data then
        # starts right after the fixed-size header.
        data_start = int(header["vox_offset"]) or _NIFTI1_HEADER_BYTES
    except Exception as e:
        # Unreadable header — corrupt, empty, or not a NIfTI at all.
        logger.debug("cannot read NIfTI header of %s: %s", path, e)
        return False

    expected = data_start + data_bytes

    if path.name.endswith(".gz"):
        if expected >= _ISIZE_MODULUS:
            logger.warning(
                "%s is larger than gzip's 4 GiB ISIZE field can describe — "
                "cannot verify completeness, assuming complete", path,
            )
            return True
        try:
            with open(path, "rb") as f:
                f.seek(-4, 2)
                isize = struct.unpack("<I", f.read(4))[0]
        except (OSError, struct.error) as e:
            logger.debug("cannot read gzip trailer of %s: %s", path, e)
            return False
        complete = expected <= isize <= expected + _EXTENSION_SLACK_BYTES
    else:
        complete = path.stat().st_size >= expected

    if not complete:
        logger.warning("incomplete NIfTI (truncated write?): %s", path)
    return complete
