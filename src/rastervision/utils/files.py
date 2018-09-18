import os
import shutil
from urllib.parse import urlparse
import tempfile
from threading import Timer
from pathlib import Path

from google.protobuf import json_format

from rastervision.filesystem.filesystem import ProtobufParseException
from rastervision.filesystem.filesystem import FileSystem
from rastervision.filesystem.local_filesystem import make_dir


def get_local_path(uri, download_dir, fs=None):
    """Convert a URI into a corresponding local path.

    If a uri is local, return it. If it's remote, we generate a path for it
    within download_dir. For an S3 path of form s3://<bucket>/<key>, the path
    is <download_dir>/s3/<bucket>/<key>.

    Args:
        uri: (string) URI of file
        download_dir: (string) path to directory
        fs: Optional FileSystem to use

    Returns:
        (string) a local path
    """
    if uri is None:
        return None

    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    path = fs.local_path(uri, download_dir)

    return path


def sync_dir(src_dir_uri, dest_dir_uri, delete=False, fs=None):
    """Synchronize a local and remote directory.

    Transfers files from source to destination directories so that the
    destination has all the source files. If delete is True, also delete
    files in the destination to match those in the source directory.

    Args:
        src_dir_uri: (string) URI of source directory
        dest_dir_uri: (string) URI of destination directory
        delete: (bool)
        fs: Optional FileSystem to use
    """
    if not fs:
        fs = FileSystem.get_file_system(dest_dir_uri, 'w')
    fs.sync_dir(src_dir_uri, dest_dir_uri, delete=delete)


def start_sync(src_dir_uri, dest_dir_uri, sync_interval=600, fs=None):
    """Start syncing a directory on a schedule.

    Calls sync_dir on a schedule.

    Args:
        src_dir_uri: (string) URI of source directory
        dest_dir_uri: (string) URI of destination directory
        sync_interval: (int) period in seconds for syncing
        fs:  Optional FileSystem to use
    """

    def _sync_dir(delete=True):
        sync_dir(src_dir_uri, dest_dir_uri, delete=delete, fs=fs)
        thread = Timer(sync_interval, _sync_dir)
        thread.daemon = True
        thread.start()

    if urlparse(dest_dir_uri).scheme == 's3':
        # On first sync, we don't want to delete files on S3 to match
        # the contents of output_dir since there's nothing there yet.
        _sync_dir(delete=False)


def download_if_needed(uri, download_dir, fs=None):
    """Download a file into a directory if it's remote.

    If uri is local, there is no need to download the file.

    Args:
        uri: (string) URI of file
        download_dir: (string) local directory to download file into
        fs: Optional FileSystem to use.

    Returns:
        (string) path to local file

    Raises:
        NotReadableError if URI cannot be read from
    """
    if uri is None:
        return None

    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')

    path = get_local_path(uri, download_dir, fs=fs)
    make_dir(path, use_dirname=True)

    print('Downloading {} to {}'.format(uri, path))

    fs.copy_from(uri, path)

    return path


def download_or_copy(uri, target_dir, fs=None):
    """Downloads or copies a file to a directory

    Args:
       uri: (string) URI of file
       target_dir: (string) local directory to copy file to
       fs: Optional FileSystem to use
    """
    local_path = download_if_needed(uri, target_dir, fs=fs)
    shutil.copy(local_path, target_dir)
    return local_path


def file_exists(uri, fs=None):
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.file_exists(uri)


def upload_or_copy(src_path, dst_uri, fs=None):
    """Upload a file if the destination is remote.

    If dst_uri is local, the file is copied.

    Args:
        src_path: (string) path to source file
        dst_uri: (string) URI of destination for file
        fs: Optional FileSystem to use
    Raises:
        NotWritableError if URI cannot be written to
    """
    if dst_uri is None:
        return

    if not (os.path.isfile(src_path) or os.path.isdir(src_path)):
        raise Exception('{} does not exist.'.format(src_path))

    print('Uploading {} to {}'.format(src_path, dst_uri))

    if not fs:
        fs = FileSystem.get_file_system(dst_uri, 'w')
    fs.copy_to(src_path, dst_uri)


def file_to_str(uri, fs=None):
    """Download contents of text file into a string.

    Args:
        uri: (string) URI of file
        fs: Optional FileSystem to use

    Returns:
        (string) with contents of text file

    Raises:
        NotReadableError if URI cannot be read from
    """
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.read_str(uri)


def str_to_file(content_str, uri, fs=None):
    """Writes string to text file.

    Args:
        content_str: string to write
        uri: (string) URI of file to write
        fs: Optional FileSystem to use

    Raise:
        NotWritableError if file_uri cannot be written
    """
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.write_str(uri, content_str)


def load_json_config(uri, message, fs=None):
    """Load a JSON-formatted protobuf config file.

    Args:
        uri: (string) URI of config file
        message: (google.protobuf.message.Message) empty protobuf message of
            to load the config into. The type needs to match the content of
            uri.
        fs: Optional FileSystem to use.

    Returns:
        the same message passed as input with fields filled in from uri

    Raises:
        ProtobufParseException if uri cannot be parsed
    """
    try:
        return json_format.Parse(file_to_str(uri, fs=fs), message)
    except json_format.ParseError as e:
        error_msg = ('Problem parsing protobuf file {}. '.format(uri) +
                     'You might need to run scripts/compile')
        raise ProtobufParseException(error_msg) from e


def save_json_config(message, uri, fs=None):
    """Save a protobuf object to a JSON file.

    Args:
        message: (google.protobuf.message.Message) protobuf message
        uri: (string) URI of JSON file to write message to
        fs: Optional FileSystem to use

    Raises:
        NotWritableError if uri cannot be written
    """
    json_str = json_format.MessageToJson(message)
    str_to_file(json_str, uri, fs=fs)


# Ensure that RV temp directory exists. We need to use a custom location for
# the temporary directory so it will be mirrored on the host file system which
# is needed for running in a Docker container with limited space on EC2.
RV_TEMP_DIR = '/opt/data/tmp/'

# find explicitly set tempdir
explicit_temp_dir = next(
    iter([
        os.environ.get(k) for k in ['TMPDIR', 'TEMP', 'TMP'] if k in os.environ
    ] + [tempfile.tempdir]))

try:
    # try to create directory
    if not os.path.exists(explicit_temp_dir):
        os.makedirs(explicit_temp_dir, exist_ok=True)
    # can we interact with directory?
    explicit_temp_dir_valid = (os.path.isdir(explicit_temp_dir) and Path.touch(
        Path(os.path.join(explicit_temp_dir, '.can_touch'))))
except Exception:
    print('Root temporary directory cannot be used: {}. Using root: {}'.format(
        explicit_temp_dir, RV_TEMP_DIR))
    tempfile.tempdir = RV_TEMP_DIR  # no guarantee this will work
    make_dir(RV_TEMP_DIR)
finally:
    # now, ensure uniqueness for this process
    # the host may be running more than one rastervision process
    RV_TEMP_DIR = tempfile.mkdtemp()
    tempfile.tempdir = RV_TEMP_DIR
    print('Temporary directory is: {}'.format(tempfile.tempdir))
