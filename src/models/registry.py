import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

import utilities
reg = utilities.Registry()

def register(name):
    def inner(func):
        reg.put(name, func)
        return func
    return inner
