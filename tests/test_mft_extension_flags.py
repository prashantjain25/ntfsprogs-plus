#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Offline MFT directory-bit regressions for extension records.

python3 tests/test_mft_extension_flags.py --build-dir .

An extension MFT record (base_mft_record != 0) only holds attributes spilled
from its base inode, so ntfs_mft_record_check() must keep its directory bit as
found, even when it carries the $INDEX_ROOT $I30 of a directory or an unnamed
$DATA attribute. The fixtures
are fresh 64 MiB mkntfs volumes; helpers come from tests/test_fsck_repairs.py.
No mounts, root privileges or external images are needed.
"""

import argparse
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import test_fsck_repairs as fixtures


BUILD_DIR = Path(__file__).resolve().parents[1]

ROOT_RECORD = 5
FIRST_USER_RECORD = 16
MFT_IN_USE = 0x0001
MFT_DIRECTORY = 0x0002
MFT_UNKNOWN_BIT = 0x0100
ATTR_STANDARD_INFORMATION = 0x10
ATTR_ATTRIBUTE_LIST = 0x20
ATTR_FILE_NAME = 0x30
ATTR_INDEX_ROOT = 0x90
ATTR_BITMAP = 0xb0
ATTR_END = 0xffffffff
FILE_ATTR_DIRECTORY = 0x10
FILE_ATTR_I30_INDEX_PRESENT = 0x10000000
RESIDENT_ATTR_IS_INDEXED = 0x01
COLLATION_FILE_NAME = 0x01
FILE_NAME_WIN32 = 0x01
INDEX_ENTRY_END = 0x0002
DIRECTORY_NAME = "EXTMFTDIR"
USA_NTFS31 = 0x30  # NTFS 3.1 record header: USA after the record number field
USA_NTFS30 = 0x2a  # NTFS 3.0 record header: USA at 0x2a, no record number
ATTR_DATA = 0x80
FILE_ATTR_ARCHIVE = 0x20
DATA_FILE_NAME = "EXTDATAFILE"
DATA_VALUE = b"extension data\n"


def align8(value):
    return (value + 7) & ~7


def mft_ref(number, sequence):
    return number | (sequence << 48)


def resident_attribute(attr_type, instance, value, name="", resident_flags=0):
    name_bytes = name.encode("utf-16le")
    value_offset = align8(24 + len(name_bytes))
    length = align8(value_offset + len(value))
    attr = bytearray(length)
    struct.pack_into("<IIBBHHHIHBB", attr, 0, attr_type, length, 0,
                     len(name_bytes) // 2, 24, 0, instance, len(value),
                     value_offset, resident_flags, 0)
    attr[24:24 + len(name_bytes)] = name_bytes
    attr[value_offset:value_offset + len(value)] = value
    return bytes(attr)


def attribute_list_entry(attr_type, reference, instance, name=""):
    name_bytes = name.encode("utf-16le")
    length = align8(26 + len(name_bytes))
    entry = bytearray(length)
    struct.pack_into("<IHBBqQH", entry, 0, attr_type, length,
                     len(name_bytes) // 2, 26, 0, reference, instance)
    entry[26:26 + len(name_bytes)] = name_bytes
    return bytes(entry)


def empty_index_root(block_size, clusters_per_block):
    """INDEX_ROOT value (48 bytes) for an empty small $I30 index."""
    value = bytearray(48)
    struct.pack_into("<IIIb", value, 0, ATTR_FILE_NAME, COLLATION_FILE_NAME,
                     block_size, clusters_per_block)
    # INDEX_HEADER at +16: entries at +16, 32 bytes used, 32 allocated, small.
    struct.pack_into("<IIIB", value, 16, 16, 32, 32, 0)
    # One END entry (16 bytes) follows the header.
    struct.pack_into("<QHHHH", value, 32, 0, 16, 0, INDEX_ENTRY_END, 0)
    return bytes(value)


def file_name_value(parent_ref, times, name, attributes):
    name_bytes = name.encode("utf-16le")
    value = bytearray(66 + len(name_bytes))
    struct.pack_into("<Q", value, 0, parent_ref)
    struct.pack_into("<QQQQ", value, 8, *times)
    struct.pack_into("<I", value, 56, attributes)
    value[64] = len(name)
    value[65] = FILE_NAME_WIN32
    value[66:] = name_bytes
    return bytes(value)


def mft_record(size, usa_ofs, number, sequence, flags, base, next_instance,
               link_count, attributes):
    """Unprotected FILE record; fixtures.Image.write_record() applies the USA."""
    count = size // 512 + 1
    attrs_offset = align8(usa_ofs + 2 * count)
    body = b"".join(attributes) + struct.pack("<II", ATTR_END, 0)
    used = attrs_offset + len(body)
    fixtures.require(used <= size, "fixture record does not fit")
    record = bytearray(size)
    record[0:4] = b"FILE"
    struct.pack_into("<HH", record, 4, usa_ofs, count)
    struct.pack_into("<HHHH", record, 16, sequence, link_count, attrs_offset, flags)
    struct.pack_into("<II", record, 24, used, size)
    struct.pack_into("<QH", record, 32, base, next_instance)
    if usa_ofs == USA_NTFS31:
        struct.pack_into("<I", record, 44, number)
    struct.pack_into("<H", record, usa_ofs, 1)
    record[attrs_offset:attrs_offset + len(body)] = body
    return record


class ExtensionFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.BUILD_DIR = BUILD_DIR
        cls.temp = tempfile.TemporaryDirectory(prefix="ntfsck-mft-flags-")
        cls.directory = Path(cls.temp.name)
        cls.base = cls.directory / "base.img"
        with cls.base.open("wb") as stream:
            stream.truncate(fixtures.IMAGE_SIZE)
        result = fixtures.command("mkntfs", "-F", "-Q", cls.base)
        fixtures.require(result.returncode == 0, result.stderr.decode(errors="replace"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.path = self.directory / "case.img"
        shutil.copyfile(self.base, self.path)

    def run_ntfsck(self, *options):
        run = fixtures.command("ntfsck", *options, self.path)
        return run.returncode, (run.stdout + run.stderr).decode(errors="replace")

    def disk_flags(self, number):
        return fixtures.u16(fixtures.Image(self.path).record(number), 22)

    @staticmethod
    def attribute_value(record, offset):
        fixtures.require(record[offset + 8] == 0, "fixture attribute is not resident")
        length = fixtures.u32(record, offset + 16)
        start = offset + fixtures.u16(record, offset + 20)
        return bytes(record[start:start + length])

    @staticmethod
    def mft_bitmap_bit(fixture, number):
        record = fixture.record(0)
        offset = fixture.attr(record, ATTR_BITMAP)
        index = number // 8
        if record[offset + 8]:
            value = fixture.read_stream(fixture.runs(record, offset), index, 1)[0]
        else:
            value = record[offset + fixtures.u16(record, offset + 20) + index]
        return bool(value & (1 << (number % 8)))

    @staticmethod
    def set_mft_bitmap_bit(fixture, number):
        record = fixture.record(0)
        offset = fixture.attr(record, ATTR_BITMAP)
        index, mask = number // 8, 1 << (number % 8)
        if record[offset + 8]:
            runs = fixture.runs(record, offset)
            value = fixture.read_stream(runs, index, 1)[0]
            fixture.write_stream(runs, index, bytes([value | mask]))
            return
        record[offset + fixtures.u16(record, offset + 20) + index] |= mask
        fixture.write_record(0, record)
        # $MFTMirr keeps a copy of record 0, so update it as well.
        raw = fixture.read_stream(fixture.mft_runs, 0, fixture.record_size)
        fixture.write(fixture.mirror_offset, raw)

    def free_user_records(self, fixture, count):
        found = []
        limit = fixture.mft_size // fixture.record_size
        for number in range(FIRST_USER_RECORD, limit):
            if not self.mft_bitmap_bit(fixture, number):
                found.append(number)
                if len(found) == count:
                    return found
        raise AssertionError("fixture has too few free user MFT records")

    def build_directory(self, fixture, layout="ntfs31", extension_flags=MFT_IN_USE):
        """Make base record B, a directory whose $I30 root lives in extension E.

        B lists its attributes in an $ATTRIBUTE_LIST that names E for the
        $INDEX_ROOT, which keeps ntfsck_verify_extent_records() from treating E
        as unreferenced. B is not linked from the root directory; the orphan
        pass relinks it, which is ordinary ntfsck behaviour and does not touch E.
        Returns (base_number, extension_number).
        """
        root = fixture.record(ROOT_RECORD)
        root_ref = mft_ref(ROOT_RECORD, fixtures.u16(root, 16))
        si = bytearray(self.attribute_value(
            root, fixture.attr(root, ATTR_STANDARD_INFORMATION)))
        index_root = self.attribute_value(
            root, fixture.attr(root, ATTR_INDEX_ROOT, "$I30"))
        fixtures.require(len(si) >= 36 and len(index_root) >= 32,
                         "unexpected root attribute values")
        dir_attributes = FILE_ATTR_DIRECTORY | FILE_ATTR_I30_INDEX_PRESENT
        struct.pack_into("<I", si, 32, dir_attributes)
        times = struct.unpack_from("<QQQQ", si, 0)
        # INDEX_ROOT: type +0, collation +4, index_block_size +8, clusters +12.
        block_size = fixtures.u32(index_root, 8)
        clusters = struct.unpack_from("<b", index_root, 12)[0]

        base_no, ext_no = self.free_user_records(fixture, 2)
        base_ref = mft_ref(base_no, 1)
        ext_ref = mft_ref(ext_no, 1)
        # Entries in type order, without an entry for $ATTRIBUTE_LIST itself.
        attribute_list = b"".join([
            attribute_list_entry(ATTR_STANDARD_INFORMATION, base_ref, 0),
            attribute_list_entry(ATTR_FILE_NAME, base_ref, 1),
            attribute_list_entry(ATTR_INDEX_ROOT, ext_ref, 3, "$I30"),
        ])
        base_attrs = [
            resident_attribute(ATTR_STANDARD_INFORMATION, 0, bytes(si)),
            resident_attribute(ATTR_ATTRIBUTE_LIST, 2, attribute_list),
            resident_attribute(
                ATTR_FILE_NAME, 1,
                file_name_value(root_ref, times, DIRECTORY_NAME, dir_attributes),
                resident_flags=RESIDENT_ATTR_IS_INDEXED),
        ]
        extension_attrs = [
            resident_attribute(ATTR_INDEX_ROOT, 3,
                               empty_index_root(block_size, clusters), "$I30"),
        ]
        usa = USA_NTFS30 if layout == "ntfs30" else USA_NTFS31
        base = mft_record(fixture.record_size, USA_NTFS31, base_no, 1,
                          MFT_IN_USE | MFT_DIRECTORY, 0, 4, 1, base_attrs)
        extension = mft_record(fixture.record_size, usa, ext_no, 1,
                               extension_flags, base_ref, 4, 0, extension_attrs)
        fixture.write_record(base_no, base)
        fixture.write_record(ext_no, extension)
        self.set_mft_bitmap_bit(fixture, base_no)
        self.set_mft_bitmap_bit(fixture, ext_no)
        return base_no, ext_no

    def build_data_extension(self, fixture, layout="ntfs31",
                             extension_flags=MFT_IN_USE | MFT_DIRECTORY):
        """Make base record B, a regular file whose unnamed $DATA lives in extension E.

        E carries flags 0x0003 but no $I30 index, so only its unnamed $DATA can
        drive the check. B lists its attributes in an $ATTRIBUTE_LIST that names
        E for the unnamed $DATA. B is not linked from the root directory; the
        orphan pass relinks it, as in build_directory().
        Returns (base_number, extension_number).
        """
        root = fixture.record(ROOT_RECORD)
        root_ref = mft_ref(ROOT_RECORD, fixtures.u16(root, 16))
        si = bytearray(self.attribute_value(
            root, fixture.attr(root, ATTR_STANDARD_INFORMATION)))
        fixtures.require(len(si) >= 36, "unexpected root attribute values")
        struct.pack_into("<I", si, 32, FILE_ATTR_ARCHIVE)
        times = struct.unpack_from("<QQQQ", si, 0)

        base_no, ext_no = self.free_user_records(fixture, 2)
        base_ref = mft_ref(base_no, 1)
        ext_ref = mft_ref(ext_no, 1)
        # Entries in type order: $STANDARD_INFORMATION, $FILE_NAME, $DATA.
        attribute_list = b"".join([
            attribute_list_entry(ATTR_STANDARD_INFORMATION, base_ref, 0),
            attribute_list_entry(ATTR_FILE_NAME, base_ref, 1),
            attribute_list_entry(ATTR_DATA, ext_ref, 3),
        ])
        base_attrs = [
            resident_attribute(ATTR_STANDARD_INFORMATION, 0, bytes(si)),
            resident_attribute(ATTR_ATTRIBUTE_LIST, 2, attribute_list),
            resident_attribute(
                ATTR_FILE_NAME, 1,
                file_name_value(root_ref, times, DATA_FILE_NAME, FILE_ATTR_ARCHIVE)),
        ]
        extension_attrs = [
            resident_attribute(ATTR_DATA, 3, DATA_VALUE),
        ]
        usa = USA_NTFS30 if layout == "ntfs30" else USA_NTFS31
        base = mft_record(fixture.record_size, USA_NTFS31, base_no, 1,
                          MFT_IN_USE, 0, 4, 1, base_attrs)
        extension = mft_record(fixture.record_size, usa, ext_no, 1,
                               extension_flags, base_ref, 4, 0, extension_attrs)
        fixture.write_record(base_no, base)
        fixture.write_record(ext_no, extension)
        self.set_mft_bitmap_bit(fixture, base_no)
        self.set_mft_bitmap_bit(fixture, ext_no)
        return base_no, ext_no

    def test_extension_record_keeps_directory_bit(self):
        """Regression: E is 0x0001 before ntfsck and must stay 0x0001 after.

        Before the fix ntfs_mft_record_check() derived DIRECTORY from the
        $INDEX_ROOT $I30 in E and rewrote its flags 0x1 -> 0x3.
        """
        for layout in ("ntfs31", "ntfs30"):
            with self.subTest(layout=layout):
                shutil.copyfile(self.base, self.path)
                _, extension = self.build_directory(fixtures.Image(self.path), layout)
                code, output = self.run_ntfsck("-a")
                self.assertIn(code, (0, 1), output)
                self.assertEqual(self.disk_flags(extension), MFT_IN_USE, output)
                self.assertNotIn(f"Inode({extension}): MFT flags are corrupted", output)

    def test_extension_data_record_keeps_directory_bit(self):
        """Regression: E holds only unnamed $DATA and flags 0x0003; it must stay 0x0003.

        Before the fix ntfs_mft_record_check() took the unnamed $DATA in E as the
        mark of a file and cleared the directory bit (0x0003 -> 0x0001). With the
        fix E keeps the bit it has, and the check does not report E.
        """
        for layout in ("ntfs31", "ntfs30"):
            with self.subTest(layout=layout):
                shutil.copyfile(self.base, self.path)
                _, extension = self.build_data_extension(fixtures.Image(self.path), layout)
                code, output = self.run_ntfsck("-a")
                self.assertIn(code, (0, 1), output)
                self.assertEqual(self.disk_flags(extension), MFT_IN_USE | MFT_DIRECTORY, output)
                self.assertNotIn(f"Inode({extension}): MFT flags are corrupted", output)

    def test_extension_flag_check_is_reached(self):
        """Control: the same E with an unknown bit is reported by old and new code.

        This shows the fixture really reaches ntfs_mft_record_check() on E, so the
        negative test above cannot pass just because ntfsck never looked at E.
        Both versions clear 0x0100; only the DIRECTORY bit differs.
        """
        for layout in ("ntfs31", "ntfs30"):
            with self.subTest(layout=layout):
                shutil.copyfile(self.base, self.path)
                _, extension = self.build_directory(
                    fixtures.Image(self.path), layout, MFT_IN_USE | MFT_UNKNOWN_BIT)
                code, output = self.run_ntfsck("-a")
                self.assertEqual(code, 1, output)
                self.assertIn(f"Inode({extension}): MFT flags are corrupted", output)
                flags = self.disk_flags(extension)
                self.assertEqual(flags & MFT_UNKNOWN_BIT, 0, output)
                self.assertEqual(flags & MFT_IN_USE, MFT_IN_USE, output)

    def test_base_directory_flag_is_restored(self):
        """Positive control: base directory root (record 5) regains DIRECTORY.

        Record 5 is the fresh mkntfs root with its own resident $INDEX_ROOT $I30,
        so the base-record derivation must still restore 0x0003 from 0x0001.
        The image is otherwise clean, so the only repair is this flag.
        """
        fixture = fixtures.Image(self.path)
        record = fixture.record(ROOT_RECORD)
        original = fixtures.u16(record, 22)
        self.assertTrue(original & MFT_DIRECTORY, "root must start as a directory")
        self.assertEqual(fixtures.u64(record, 32), 0, "root must be a base record")
        fixture.attr(record, ATTR_INDEX_ROOT, "$I30")
        cleared = original & ~MFT_DIRECTORY
        struct.pack_into("<H", record, 22, cleared)
        fixture.write_record(ROOT_RECORD, record)

        code, output = self.run_ntfsck("-a")
        self.assertEqual(code, 1, output)
        self.assertIn(
            f"Inode({ROOT_RECORD}): MFT flags are corrupted "
            f"(0x{cleared:x} <> 0x{original:x}). Fixed.", output)
        self.assertEqual(self.disk_flags(ROOT_RECORD), original, output)
        code, output = self.run_ntfsck("-n")
        self.assertEqual(code, 0, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    options, remaining = parser.parse_known_args()
    BUILD_DIR = options.build_dir.resolve()
    unittest.main(argv=[str(Path(__file__)), *remaining])
