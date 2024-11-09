from pathlib import Path
import os


def get_approximate_csv_file_lines(file_path: Path) -> int:
    """Method to get approximate number of lines in csv file"""
    max_check_lines: int = 1000
    read_lines: int = 0
    red_data: str = ""
    with open(file_path) as file:
        for _ in range(max_check_lines):
            red_data += file.readline()
            read_lines += 1
        max_check_lines = read_lines if read_lines < max_check_lines else max_check_lines
        n_lines = int(os.path.getsize(file_path) / (len(red_data.encode('utf-8')) / float(max_check_lines)))
    return n_lines
