from pathlib import Path
import glob

def test_a_imports():
    """
    Please don't import sklearn or scipy.stats to solve any of the problems in this assignment. 
    If you fail this test, we will give you a zero for this assignment, regardless of how
    sklearn or scipy.stats was used in your code.

    the 'a' in the file name is so this test is run first on a clean Python interpreter.
    """
    import sys
    import src

    source_dir: Path = Path(__file__).parent.parent / 'src'
    source_files: list = [Path(p) for p in glob.glob(str(source_dir / "*.py"))]

    for f in source_files:
        assert not is_imported(f, 'sklearn'), "use of sklearn is not permitted in this assignment."
        assert not is_imported(f, 'scipy'), "use of scipy is not permitted in this assignment."

def is_imported(filepath: Path, pkg_name: str) -> bool:
    """checks if a package has been imported in a file
    """

    imported = False
    with open(filepath, 'r') as f:
        content = f.read()
        tokens = content.split()

    check_indexes = []
    for idx, t in enumerate(tokens):
        if pkg_name in t:
            check_indexes.append(idx)


    for idx in check_indexes:
        try:
            assert idx > 0
            assert idx < len(tokens) - 1

            if tokens[idx-1] ==  'import': 
                imported = True
            elif tokens[idx-1] == 'from' and tokens[idx+1] == "import":
                imported = True
        except ValueError:
            pass

    return imported