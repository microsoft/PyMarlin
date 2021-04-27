'''
Stats package.
'''
from .basic_stats import BasicStats, StatInitArguments
#singleton object
global_stats = BasicStats(StatInitArguments(), writers=[])
