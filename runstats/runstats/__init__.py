"""
Python RunStats - Online Statistics and Regression
==================================================

"""

try:
    from .fast import Statistics, Regression
    __compiled__ = True
except ImportError:
    from .core import Statistics, Regression
    __compiled__ = False

__title__ = 'runstats'
__version__ = '1.8.0'
__build__ = 0x010800
__author__ = 'Grant Jenks'
__license__ = 'Apache 2.0'
__copyright__ = '2013-2019, Grant Jenks'
