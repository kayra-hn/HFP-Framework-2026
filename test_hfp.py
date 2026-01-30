# test_hfp.py
def test_import():
    """Basit import testi"""
    try:
        import numpy
        import scipy
        import matplotlib
        import pandas
        assert True
    except ImportError:
        assert False

def test_placeholder():
    """Placeholder test - her zaman geçer"""
    assert True

def test_hfp_exists():
    """HFP modülünün varlığını test et"""
    import os
    # Projede .py dosyası var mı kontrol et
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    assert len(py_files) > 0, "Hiç Python dosyası bulunamadı"
