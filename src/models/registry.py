reg = {}

def register(name):
    def inner(func):
        assert(name not in reg)
        reg[name] = func
        return func
    return inner
