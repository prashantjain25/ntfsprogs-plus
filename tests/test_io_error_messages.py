#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""I/O error messages must name the offset and byte count of the failed request.

An LD_PRELOAD shim makes one pread or pwrite on a fresh 64 MiB regular-file
image fail with EIO. The shim matches the image by device and inode, so no
other file or device is touched. Linux only.

Run after building: python3 tests/test_io_error_messages.py --build-dir .
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import test_fsck_repairs as fixtures


BUILD_DIR = Path(__file__).resolve().parents[1]
VOLUME_RECORD = 3           # $Volume
VOLUME_INFORMATION = 0x70   # unnamed, resident
FLAGS_OFFSET = 10           # reserved (8) + major (1) + minor (1)
VOLUME_IS_DIRTY = 0x0001

SHIM = r'''
#define _GNU_SOURCE
#define _LARGEFILE64_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

/* True if fd refers to the image named by IOERR_DEV and IOERR_INO. */
static int is_image(int fd)
{
    struct stat st;
    const char *dev = getenv("IOERR_DEV"), *ino = getenv("IOERR_INO");

    return dev && ino && !fstat(fd, &st) && S_ISREG(st.st_mode) &&
        (unsigned long long)st.st_dev == strtoull(dev, NULL, 10) &&
        (unsigned long long)st.st_ino == strtoull(ino, NULL, 10);
}

/* Fail only the first call of kind op at or past IOERR_MIN_OFFSET. */
static int should_fail(const char *op, off_t off)
{
    static int done;
    const char *want = getenv("IOERR_OP"), *min = getenv("IOERR_MIN_OFFSET");

    if (done || !want || strcmp(want, op) || !min ||
        off < (off_t)strtoll(min, NULL, 10))
        return 0;
    done = 1;
    return 1;
}

static ssize_t shim_pread(int fd, void *buf, size_t n, off_t off)
{
    if (is_image(fd) && should_fail("pread", off)) {
        fprintf(stderr, "IOERR_FAULT pread offset=%lld size=%zu\n",
                (long long)off, n);
        errno = EIO;
        return -1;
    }
    return syscall(SYS_pread64, fd, buf, n, off);
}

static ssize_t shim_pwrite(int fd, const void *buf, size_t n, off_t off)
{
    if (is_image(fd) && should_fail("pwrite", off)) {
        fprintf(stderr, "IOERR_FAULT pwrite offset=%lld size=%zu\n",
                (long long)off, n);
        errno = EIO;
        return -1;
    }
    return syscall(SYS_pwrite64, fd, buf, n, off);
}

ssize_t pread(int fd, void *buf, size_t n, off_t off)
{
    return shim_pread(fd, buf, n, off);
}

ssize_t pread64(int fd, void *buf, size_t n, off64_t off)
{
    return shim_pread(fd, buf, n, (off_t)off);
}

ssize_t pwrite(int fd, const void *buf, size_t n, off_t off)
{
    return shim_pwrite(fd, buf, n, off);
}

ssize_t pwrite64(int fd, const void *buf, size_t n, off64_t off)
{
    return shim_pwrite(fd, buf, n, (off_t)off);
}
'''


@unittest.skipUnless(sys.platform.startswith("linux"),
                     "LD_PRELOAD fault shim needs Linux")
class IoErrorMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.BUILD_DIR = BUILD_DIR
        cls.workspace = tempfile.TemporaryDirectory(prefix="ntfsck-ioerr-")
        cls.directory = Path(cls.workspace.name)
        cls.pristine = cls.directory / "pristine.img"
        with cls.pristine.open("wb") as stream:
            stream.truncate(fixtures.IMAGE_SIZE)
        result = fixtures.command("mkntfs", "-F", "-Q", "-s", "512", cls.pristine)
        fixtures.require(result.returncode == 0,
                         result.stderr.decode(errors="replace"))
        source = cls.directory / "ioerr.c"
        source.write_text(SHIM)
        cls.shim = cls.directory / "ioerr.so"
        result = subprocess.run(
            ["cc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-O2",
             "-U_FORTIFY_SOURCE", "-o", str(cls.shim), str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
            check=False)
        fixtures.require(result.returncode == 0,
                         result.stdout.decode(errors="replace"))

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.image = self.directory / "case.img"
        shutil.copyfile(self.pristine, self.image)

    def set_dirty(self):
        """Set VOLUME_IS_DIRTY in both copies of the $Volume record."""
        fixture = fixtures.Image(self.image)
        record = fixture.record(VOLUME_RECORD)
        offset = fixture.attr(record, VOLUME_INFORMATION)
        flags = offset + fixtures.u16(record, offset + 20) + FLAGS_OFFSET
        struct.pack_into("<H", record, flags, VOLUME_IS_DIRTY)
        fixture.write_record(VOLUME_RECORD, record)
        raw = fixture.read_stream(fixture.mft_runs,
                                  VOLUME_RECORD * fixture.record_size,
                                  fixture.record_size)
        fixture.write(fixture.mirror_offset + VOLUME_RECORD * fixture.record_size,
                      raw)

    def run_with_fault(self, args, op):
        """Run ntfsck with the first failing op at or past the MFT offset.

        Returns (exit status, output, faulted offset, faulted size).
        """
        fixture = fixtures.Image(self.image)
        st = self.image.stat()
        env = os.environ.copy()
        env.update(LC_ALL="C", LD_PRELOAD=str(self.shim),
                   IOERR_DEV=str(st.st_dev), IOERR_INO=str(st.st_ino),
                   IOERR_OP=op, IOERR_MIN_OFFSET=str(fixture.mft_offset))
        result = subprocess.run(
            [str(BUILD_DIR / "src" / "ntfsck"), *args, str(self.image)],
            input=b"", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, timeout=60, check=False)
        output = (result.stdout + result.stderr).decode(errors="replace")
        match = re.search(r"IOERR_FAULT %s offset=(\d+) size=(\d+)" % op, output)
        fixtures.require(match, "shim did not inject a fault:\n" + output)
        return result.returncode, output, int(match[1]), int(match[2])

    def test_failed_pread_names_offset_and_size(self):
        code, output, offset, size = self.run_with_fault(["-n"], "pread")
        self.assertEqual(code, 8, output)
        self.assertIn("Failed to pread device", output)
        self.assertIn(f"at offset {offset} for {size} bytes", output)

    def test_failed_pwrite_names_offset_and_size(self):
        self.set_dirty()
        code, output, offset, size = self.run_with_fault(["-a"], "pwrite")
        self.assertEqual(code, 8, output)
        self.assertIn("Failed to pwrite device", output)
        self.assertIn(f"at offset {offset} for {size} bytes", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    options, remaining = parser.parse_known_args()
    BUILD_DIR = options.build_dir.resolve()
    unittest.main(argv=[str(Path(__file__)), *remaining])
