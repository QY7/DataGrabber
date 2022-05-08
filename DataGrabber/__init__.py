import sys,os
print(f"{os.path.dirname(__file__)}")
sys.path.append(os.path.dirname(__file__))
from .datagrabber import datagrabber