def test_imports():
    """
    Please don't import scipy.stats to solve any of the problems in this assignment. 
    If you fail this test, we will give you a zero for this assignment, regardless of how
    scipy.stats was used in your code.

    the 'a' in the file name is so this test is run first on a clean Python interpreter.
    """
    import sys
    import src

    assert 'scipy.stats' not in sys.modules.keys()