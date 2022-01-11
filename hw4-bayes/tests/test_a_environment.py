def test_imports():
    """
    Please don't import any of the listed packages below to solve any of the problems in this assignment. 
    If you fail this test, we will give you a zero for this assignment, regardless of how
    scipy.stats was used in your code.

    the 'a' in the file name is so this test is run first on a clean Python interpreter.
    """
    import sys
    import src
    disallowed_imports = ['scipy.stats', 'scipy.special', 'sklearn']
    for imp in disallowed_imports:
        assert imp not in sys.modules.keys()
