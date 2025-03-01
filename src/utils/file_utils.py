# file_utils.py - 文件目录处理工具模块

"""
文件说明：
    这个文件提供了GCG_Quant系统中处理文件和目录的工具函数。
    包含文件读写、目录管理、文件搜索等常用功能。
    这些工具函数简化了系统中的文件操作，提高代码可读性和可维护性。

学习目标：
    1. 了解Python中文件和目录操作的常用模式
    2. 学习如何安全地处理文件路径和创建目录
    3. 掌握文件搜索和过滤的技术
"""

import os
import shutil
import glob
import json
import csv
import pickle
import zipfile
import gzip
import datetime
import re
from pathlib import Path
from typing import (
    List,
    Dict,
    Any,
    Union,
    Optional,
    Callable,
    Tuple,
    Set,
    BinaryIO,
    TextIO,
    Iterator,
)


def ensure_directory(directory: str) -> bool:
    """
    确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径

    Returns:
        bool: 操作是否成功
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建目录失败: {str(e)}")
        return False


def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名

    Args:
        file_path: 文件路径

    Returns:
        str: 文件扩展名（不包含点）
    """
    return os.path.splitext(file_path)[1][1:].lower()


def is_file_type(file_path: str, extension: str) -> bool:
    """
    检查文件是否为指定类型

    Args:
        file_path: 文件路径
        extension: 扩展名（不包含点）

    Returns:
        bool: 是否为指定类型
    """
    if not extension:
        return False

    # 确保扩展名不包含点
    if extension.startswith("."):
        extension = extension[1:]

    return get_file_extension(file_path).lower() == extension.lower()


def get_directory_size(directory: str) -> int:
    """
    获取目录大小（字节）

    Args:
        directory: 目录路径

    Returns:
        int: 目录大小（字节）
    """
    total_size = 0

    # 使用os.walk遍历目录
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)

    return total_size


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        str: 格式化后的文件大小
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def list_files(
    directory: str, pattern: str = "*", recursive: bool = False
) -> List[str]:
    """
    列出目录中的文件

    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        recursive: 是否递归子目录

    Returns:
        List[str]: 文件路径列表
    """
    if recursive:
        pattern_path = os.path.join(directory, "**", pattern)
        return [f for f in glob.glob(pattern_path, recursive=True) if os.path.isfile(f)]
    else:
        pattern_path = os.path.join(directory, pattern)
        return [f for f in glob.glob(pattern_path) if os.path.isfile(f)]


def list_directories(directory: str, recursive: bool = False) -> List[str]:
    """
    列出目录中的子目录

    Args:
        directory: 目录路径
        recursive: 是否递归子目录

    Returns:
        List[str]: 子目录路径列表
    """
    if recursive:
        result = []
        for dirpath, dirnames, _ in os.walk(directory):
            result.extend([os.path.join(dirpath, d) for d in dirnames])
        return result
    else:
        return [
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]


def copy_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    复制文件

    Args:
        source: 源文件路径
        destination: 目标文件路径
        overwrite: 是否覆盖已存在的文件

    Returns:
        bool: 操作是否成功
    """
    try:
        # 检查源文件是否存在
        if not os.path.isfile(source):
            print(f"源文件不存在: {source}")
            return False

        # 检查目标文件是否已存在
        if os.path.exists(destination) and not overwrite:
            print(f"目标文件已存在: {destination}")
            return False

        # 确保目标目录存在
        destination_dir = os.path.dirname(destination)
        ensure_directory(destination_dir)

        # 复制文件
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"复制文件失败: {str(e)}")
        return False


def move_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    移动文件

    Args:
        source: 源文件路径
        destination: 目标文件路径
        overwrite: 是否覆盖已存在的文件

    Returns:
        bool: 操作是否成功
    """
    try:
        # 检查源文件是否存在
        if not os.path.isfile(source):
            print(f"源文件不存在: {source}")
            return False

        # 检查目标文件是否已存在
        if os.path.exists(destination) and not overwrite:
            print(f"目标文件已存在: {destination}")
            return False

        # 确保目标目录存在
        destination_dir = os.path.dirname(destination)
        ensure_directory(destination_dir)

        # 移动文件
        shutil.move(source, destination)
        return True
    except Exception as e:
        print(f"移动文件失败: {str(e)}")
        return False


def delete_file(file_path: str) -> bool:
    """
    删除文件

    Args:
        file_path: 文件路径

    Returns:
        bool: 操作是否成功
    """
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            return True
        else:
            print(f"文件不存在: {file_path}")
            return False
    except Exception as e:
        print(f"删除文件失败: {str(e)}")
        return False


def delete_directory(directory: str, recursive: bool = False) -> bool:
    """
    删除目录

    Args:
        directory: 目录路径
        recursive: 是否递归删除子目录和文件

    Returns:
        bool: 操作是否成功
    """
    try:
        if not os.path.isdir(directory):
            print(f"目录不存在: {directory}")
            return False

        if recursive:
            shutil.rmtree(directory)
        else:
            os.rmdir(directory)

        return True
    except Exception as e:
        print(f"删除目录失败: {str(e)}")
        return False


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    获取文件信息

    Args:
        file_path: 文件路径

    Returns:
        Dict[str, Any]: 文件信息字典
    """
    if not os.path.isfile(file_path):
        return {"exists": False}

    try:
        stat = os.stat(file_path)
        return {
            "exists": True,
            "size": stat.st_size,
            "formatted_size": format_file_size(stat.st_size),
            "created": datetime.datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime),
            "accessed": datetime.datetime.fromtimestamp(stat.st_atime),
            "is_hidden": os.path.basename(file_path).startswith("."),
            "path": os.path.abspath(file_path),
            "directory": os.path.dirname(os.path.abspath(file_path)),
            "name": os.path.basename(file_path),
            "extension": get_file_extension(file_path),
        }
    except Exception as e:
        print(f"获取文件信息失败: {str(e)}")
        return {"exists": True, "error": str(e)}


def find_files(
    directory: str,
    name_pattern: Optional[str] = None,
    extension: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    min_modified: Optional[datetime.datetime] = None,
    max_modified: Optional[datetime.datetime] = None,
    recursive: bool = True,
) -> List[str]:
    """
    查找文件

    Args:
        directory: 目录路径
        name_pattern: 文件名匹配模式（正则表达式）
        extension: 文件扩展名筛选
        min_size: 最小文件大小（字节）
        max_size: 最大文件大小（字节）
        min_modified: 最早修改时间
        max_modified: 最晚修改时间
        recursive: 是否递归子目录

    Returns:
        List[str]: 匹配的文件路径列表
    """
    result = []

    # 编译正则表达式
    name_regex = re.compile(name_pattern) if name_pattern else None

    # 规范扩展名
    if extension and extension.startswith("."):
        extension = extension[1:]

    # 遍历目录
    for root, _, files in os.walk(directory):
        # 如果不递归且不是起始目录，跳过
        if not recursive and root != directory:
            continue

        for file in files:
            file_path = os.path.join(root, file)

            # 检查文件名是否匹配
            if name_regex and not name_regex.search(file):
                continue

            # 检查扩展名是否匹配
            if extension and not file.lower().endswith(f".{extension.lower()}"):
                continue

            # 检查文件大小
            if min_size is not None or max_size is not None:
                try:
                    file_size = os.path.getsize(file_path)
                    if min_size is not None and file_size < min_size:
                        continue
                    if max_size is not None and file_size > max_size:
                        continue
                except:
                    continue

            # 检查修改时间
            if min_modified is not None or max_modified is not None:
                try:
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    if min_modified is not None and mtime < min_modified:
                        continue
                    if max_modified is not None and mtime > max_modified:
                        continue
                except:
                    continue

            result.append(file_path)

    return result


def read_text_file(file_path: str, encoding: str = "utf-8") -> Optional[str]:
    """
    读取文本文件

    Args:
        file_path: 文件路径
        encoding: 文件编码

    Returns:
        Optional[str]: 文件内容，如果读取失败则返回None
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        print(f"读取文本文件失败: {str(e)}")
        return None


def write_text_file(
    file_path: str, content: str, encoding: str = "utf-8", append: bool = False
) -> bool:
    """
    写入文本文件

    Args:
        file_path: 文件路径
        content: 文件内容
        encoding: 文件编码
        append: 是否追加模式

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_directory(directory)

        # 写入文件
        mode = "a" if append else "w"
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"写入文本文件失败: {str(e)}")
        return False


def read_binary_file(file_path: str) -> Optional[bytes]:
    """
    读取二进制文件

    Args:
        file_path: 文件路径

    Returns:
        Optional[bytes]: 文件内容，如果读取失败则返回None
    """
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"读取二进制文件失败: {str(e)}")
        return None


def write_binary_file(file_path: str, content: bytes, append: bool = False) -> bool:
    """
    写入二进制文件

    Args:
        file_path: 文件路径
        content: 文件内容
        append: 是否追加模式

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_directory(directory)

        # 写入文件
        mode = "ab" if append else "wb"
        with open(file_path, mode) as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"写入二进制文件失败: {str(e)}")
        return False


def read_json_file(file_path: str, encoding: str = "utf-8") -> Optional[Any]:
    """
    读取JSON文件

    Args:
        file_path: 文件路径
        encoding: 文件编码

    Returns:
        Optional[Any]: 解析后的JSON数据，如果读取或解析失败则返回None
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {str(e)}")
        return None


def write_json_file(
    file_path: str, data: Any, encoding: str = "utf-8", indent: int = 4
) -> bool:
    """
    写入JSON文件

    Args:
        file_path: 文件路径
        data: JSON数据
        encoding: 文件编码
        indent: 缩进空格数

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_directory(directory)

        # 写入文件
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent)

        return True
    except Exception as e:
        print(f"写入JSON文件失败: {str(e)}")
        return False


def read_csv_file(
    file_path: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    has_header: bool = True,
) -> Optional[List[Dict[str, str]]]:
    """
    读取CSV文件

    Args:
        file_path: 文件路径
        delimiter: 分隔符
        encoding: 文件编码
        has_header: 是否有表头

    Returns:
        Optional[List[Dict[str, str]]]: 解析后的CSV数据，如果读取或解析失败则返回None
    """
    try:
        result = []
        with open(file_path, "r", encoding=encoding, newline="") as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                result = [row for row in reader]
            else:
                reader = csv.reader(f, delimiter=delimiter)
                rows = [row for row in reader]
                if not rows:
                    return []

                # 创建默认列名 (col0, col1, ...)
                header = [f"col{i}" for i in range(len(rows[0]))]

                # 转换为字典列表
                for row in rows:
                    result.append(
                        {
                            header[i]: value
                            for i, value in enumerate(row)
                            if i < len(header)
                        }
                    )

        return result
    except Exception as e:
        print(f"读取CSV文件失败: {str(e)}")
        return None


def write_csv_file(
    file_path: str,
    data: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> bool:
    """
    写入CSV文件

    Args:
        file_path: 文件路径
        data: CSV数据（字典列表）
        fieldnames: 字段名列表，如果为None则使用第一行的所有键
        delimiter: 分隔符
        encoding: 文件编码

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_directory(directory)

        # 确定字段名
        if not fieldnames and data:
            fieldnames = list(data[0].keys())

        # 写入文件
        with open(file_path, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)

        return True
    except Exception as e:
        print(f"写入CSV文件失败: {str(e)}")
        return False


def read_pickle_file(file_path: str) -> Optional[Any]:
    """
    读取Pickle文件

    Args:
        file_path: 文件路径

    Returns:
        Optional[Any]: 解析后的Pickle数据，如果读取或解析失败则返回None
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"读取Pickle文件失败: {str(e)}")
        return None


def write_pickle_file(file_path: str, data: Any) -> bool:
    """
    写入Pickle文件

    Args:
        file_path: 文件路径
        data: Pickle数据

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_directory(directory)

        # 写入文件
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        return True
    except Exception as e:
        print(f"写入Pickle文件失败: {str(e)}")
        return False


def compress_files(
    files: List[str], zip_file: str, remove_original: bool = False
) -> bool:
    """
    压缩文件

    Args:
        files: 文件路径列表
        zip_file: 目标压缩文件路径
        remove_original: 是否删除原始文件

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(zip_file)
        ensure_directory(directory)

        # 创建压缩文件
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                if os.path.isfile(file):
                    zipf.write(file, os.path.basename(file))

        # 删除原始文件
        if remove_original:
            for file in files:
                if os.path.isfile(file):
                    os.remove(file)

        return True
    except Exception as e:
        print(f"压缩文件失败: {str(e)}")
        return False


def extract_zip(zip_file: str, extract_dir: str) -> bool:
    """
    解压缩ZIP文件

    Args:
        zip_file: ZIP文件路径
        extract_dir: 解压目标目录

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        ensure_directory(extract_dir)

        # 解压文件
        with zipfile.ZipFile(zip_file, "r") as zipf:
            zipf.extractall(extract_dir)

        return True
    except Exception as e:
        print(f"解压缩ZIP文件失败: {str(e)}")
        return False


def compress_gzip(
    file_path: str, output_path: Optional[str] = None, remove_original: bool = False
) -> bool:
    """
    使用gzip压缩文件

    Args:
        file_path: 文件路径
        output_path: 输出文件路径，如果为None则自动添加.gz后缀
        remove_original: 是否删除原始文件

    Returns:
        bool: 操作是否成功
    """
    try:
        if not os.path.isfile(file_path):
            print(f"文件不存在: {file_path}")
            return False

        if output_path is None:
            output_path = f"{file_path}.gz"

        # 确保目录存在
        directory = os.path.dirname(output_path)
        ensure_directory(directory)

        # 压缩文件
        with open(file_path, "rb") as f_in:
            with gzip.open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 删除原始文件
        if remove_original:
            os.remove(file_path)

        return True
    except Exception as e:
        print(f"使用gzip压缩文件失败: {str(e)}")
        return False


def extract_gzip(
    gzip_file: str, output_path: Optional[str] = None, remove_original: bool = False
) -> bool:
    """
    解压缩gzip文件

    Args:
        gzip_file: gzip文件路径
        output_path: 输出文件路径，如果为None则自动去除.gz后缀
        remove_original: 是否删除原始文件

    Returns:
        bool: 操作是否成功
    """
    try:
        if not os.path.isfile(gzip_file):
            print(f"文件不存在: {gzip_file}")
            return False

        if output_path is None:
            if gzip_file.endswith(".gz"):
                output_path = gzip_file[:-3]
            else:
                output_path = f"{gzip_file}.extracted"

        # 确保目录存在
        directory = os.path.dirname(output_path)
        ensure_directory(directory)

        # 解压文件
        with gzip.open(gzip_file, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 删除原始文件
        if remove_original:
            os.remove(gzip_file)

        return True
    except Exception as e:
        print(f"解压缩gzip文件失败: {str(e)}")
        return False


def get_file_hash(file_path: str, hash_type: str = "md5") -> Optional[str]:
    """
    计算文件哈希值

    Args:
        file_path: 文件路径
        hash_type: 哈希类型，可选值: 'md5', 'sha1', 'sha256'

    Returns:
        Optional[str]: 哈希值，如果计算失败则返回None
    """
    import hashlib

    if not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}")
        return None

    try:
        # 选择哈希算法
        if hash_type == "md5":
            hash_algo = hashlib.md5()
        elif hash_type == "sha1":
            hash_algo = hashlib.sha1()
        elif hash_type == "sha256":
            hash_algo = hashlib.sha256()
        else:
            print(f"不支持的哈希类型: {hash_type}")
            return None

        # 计算哈希值
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)

        return hash_algo.hexdigest()
    except Exception as e:
        print(f"计算文件哈希值失败: {str(e)}")
        return None


def create_temp_file(
    prefix: str = "", suffix: str = "", content: Optional[Union[str, bytes]] = None
) -> Optional[str]:
    """
    创建临时文件

    Args:
        prefix: 文件名前缀
        suffix: 文件名后缀
        content: 文件内容，如果为字符串则以文本模式写入，如果为bytes则以二进制模式写入

    Returns:
        Optional[str]: 临时文件路径，如果创建失败则返回None
    """
    import tempfile

    try:
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(
            prefix=prefix, suffix=suffix, delete=False
        )
        temp_file_path = temp_file.name
        temp_file.close()

        # 写入内容
        if content is not None:
            if isinstance(content, str):
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            elif isinstance(content, bytes):
                with open(temp_file_path, "wb") as f:
                    f.write(content)

        return temp_file_path
    except Exception as e:
        print(f"创建临时文件失败: {str(e)}")
        return None


def create_temp_directory(prefix: str = "") -> Optional[str]:
    """
    创建临时目录

    Args:
        prefix: 目录名前缀

    Returns:
        Optional[str]: 临时目录路径，如果创建失败则返回None
    """
    import tempfile

    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        return temp_dir
    except Exception as e:
        print(f"创建临时目录失败: {str(e)}")
        return None


def get_temp_directory() -> str:
    """
    获取系统临时目录

    Returns:
        str: 系统临时目录路径
    """
    import tempfile

    return tempfile.gettempdir()


def touch_file(file_path: str) -> bool:
    """
    创建空文件或更新文件的访问和修改时间

    Args:
        file_path: 文件路径

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_directory(directory)

        # 更新文件时间或创建空文件
        Path(file_path).touch()

        return True
    except Exception as e:
        print(f"创建或更新文件失败: {str(e)}")
        return False


def copy_directory(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    复制目录

    Args:
        source: 源目录路径
        destination: 目标目录路径
        overwrite: 是否覆盖已存在的文件

    Returns:
        bool: 操作是否成功
    """
    try:
        # 检查源目录是否存在
        if not os.path.isdir(source):
            print(f"源目录不存在: {source}")
            return False

        # 检查目标目录是否已存在
        if os.path.exists(destination) and not overwrite:
            print(f"目标目录已存在: {destination}")
            return False

        # 复制目录
        if overwrite and os.path.exists(destination):
            shutil.rmtree(destination)

        shutil.copytree(source, destination)

        return True
    except Exception as e:
        print(f"复制目录失败: {str(e)}")
        return False


def read_file_lines(file_path: str, encoding: str = "utf-8") -> Optional[List[str]]:
    """
    读取文件行

    Args:
        file_path: 文件路径
        encoding: 文件编码

    Returns:
        Optional[List[str]]: 文件行列表，如果读取失败则返回None
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.readlines()
    except Exception as e:
        print(f"读取文件行失败: {str(e)}")
        return None


def write_file_lines(
    file_path: str, lines: List[str], encoding: str = "utf-8", append: bool = False
) -> bool:
    """
    写入文件行

    Args:
        file_path: 文件路径
        lines: 文件行列表
        encoding: 文件编码
        append: 是否追加模式

    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_directory(directory)

        # 写入文件
        mode = "a" if append else "w"
        with open(file_path, mode, encoding=encoding) as f:
            f.writelines(lines)

        return True
    except Exception as e:
        print(f"写入文件行失败: {str(e)}")
        return False


def get_absolute_path(relative_path: str) -> str:
    """
    获取绝对路径

    Args:
        relative_path: 相对路径

    Returns:
        str: 绝对路径
    """
    return os.path.abspath(relative_path)


def get_relative_path(path: str, base_path: str) -> str:
    """
    获取相对路径

    Args:
        path: 路径
        base_path: 基准路径

    Returns:
        str: 相对路径
    """
    return os.path.relpath(path, base_path)


def is_path_exists(path: str) -> bool:
    """
    检查路径是否存在

    Args:
        path: 路径

    Returns:
        bool: 路径是否存在
    """
    return os.path.exists(path)


def is_file_exists(file_path: str) -> bool:
    """
    检查文件是否存在

    Args:
        file_path: 文件路径

    Returns:
        bool: 文件是否存在
    """
    return os.path.isfile(file_path)


def is_directory_exists(directory: str) -> bool:
    """
    检查目录是否存在

    Args:
        directory: 目录路径

    Returns:
        bool: 目录是否存在
    """
    return os.path.isdir(directory)


def is_path_empty(path: str) -> bool:
    """
    检查路径是否为空

    Args:
        path: 路径

    Returns:
        bool: 路径是否为空
    """
    if not os.path.exists(path):
        return True

    if os.path.isfile(path):
        return os.path.getsize(path) == 0

    return len(os.listdir(path)) == 0


# 使用示例
if __name__ == "__main__":
    # 测试目录操作
    temp_dir = create_temp_directory("test_")
    print(f"创建临时目录: {temp_dir}")

    # 测试文件操作
    temp_file = create_temp_file(prefix="test_", suffix=".txt", content="Hello, World!")
    print(f"创建临时文件: {temp_file}")

    # 读取文件
    content = read_text_file(temp_file)
    print(f"文件内容: {content}")

    # 获取文件信息
    file_info = get_file_info(temp_file)
    print(f"文件信息: {file_info}")

    # 测试JSON操作
    json_file = os.path.join(temp_dir, "test.json")
    data = {"name": "test", "value": 123}
    write_json_file(json_file, data)
    read_data = read_json_file(json_file)
    print(f"读取的JSON数据: {read_data}")

    # 测试CSV操作
    csv_file = os.path.join(temp_dir, "test.csv")
    csv_data = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
    write_csv_file(csv_file, csv_data)
    read_csv_data = read_csv_file(csv_file)
    print(f"读取的CSV数据: {read_csv_data}")

    # 测试压缩操作
    zip_file = os.path.join(temp_dir, "test.zip")
    compress_files([temp_file, json_file, csv_file], zip_file)

    extract_dir = os.path.join(temp_dir, "extracted")
    extract_zip(zip_file, extract_dir)

    # 清理
    delete_directory(temp_dir, recursive=True)
    delete_file(temp_file)
