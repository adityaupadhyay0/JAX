import os
import sys

# Load the C++ extension
sys.path.append(os.path.dirname(__file__))
from ._axe import *
from .axe import *