def test_imports():
    try :
        from my_parser import local, distant
        from data_structures import candidate, job
    except : 
        assert(False)
    else :
        assert(True)
    assert(True)