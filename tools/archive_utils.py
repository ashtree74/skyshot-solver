"""Conservative extraction of regular files/directories on Python 3.11+."""
import shutil
import tarfile
from pathlib import PurePosixPath

from data_bundle import require


def safe_relative(name):
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and ".." not in path.parts and
            "\\" not in name and ":" not in name, f"unsafe archive path: {name}")
    return path


def unpack_tar(archive, destination):
    # Do not rely on the tarfile filter API, which is absent in early Python 3.11.
    # No ownership, special modes, links or device entries are applied.
    with tarfile.open(archive) as stream:
        for member in stream:
            relative = safe_relative(member.name)
            require(member.isfile() or member.isdir(), "archive links/special files refused")
            path = destination.joinpath(*relative.parts)
            require(path.resolve().is_relative_to(destination.resolve()), "archive path escapes destination")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, path.open("xb") as out:
                    shutil.copyfileobj(source, out)
                path.chmod(0o755 if member.mode & 0o111 else 0o644)
