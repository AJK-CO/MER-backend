

import pytest

@pytest.mark.parametrize("a,b,c",[(1,1,2),(1,1,3)])
def test_func(a,b,c):
    assert (a+b)==c

def test_func2():
    assert 2==3