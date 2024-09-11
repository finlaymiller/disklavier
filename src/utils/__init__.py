import re
from rich.console import Console


console = Console(record=True, log_time_format="%m-%d %H:%M:%S.%f")


def extract_transformations(filename: str) -> tuple[str, dict[str, int]]:
    """
    Parses the filename, strips out the transformation section, and returns the modified filename
    and the transformations as a dictionary. Transformations are formatted as `_t{transpose}s{shift}`,
    where `transpose` and `shift` are integers on the range [0, 11] and [0, 7], respectively.
    They are both zero padded to two digits.

    Args:
        filename (str): The input filename with the transformation section.

    Returns:
        tuple: A tuple containing the modified filename and the transformations dictionary.
    """
    pattern = r"_(t\d+s\d+)\.mid$"  # regex to match the transformation section
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(
            "Filename format is incorrect or does not contain transformations."
        )
    transformation_str = match.group(1)
    transpose = int(transformation_str[1:3])
    shift = int(transformation_str[4:6])

    return (
        filename.replace(f"_{transformation_str}", ""),
        {"transpose": transpose, "shift": shift},
    )
