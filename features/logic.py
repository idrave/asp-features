from pathlib import Path
from typing import List
import clingo
class Logic:
    logicPath = Path(__file__).parent.absolute()
    #logicPath = Path('/home/ivan/Documents/ai/features/features')
    grammarFile = str(logicPath/'concept.lp')
    featureFile = str(logicPath/'features.lp')
    base = ('base', [])
    roles = ('roles', [])
    pruneFile = str(logicPath/'prune.lp')
    
    keepRoles = ('get_roles', [])
    @staticmethod
    def index_role(start):
        return ('index_role', [start])
    @staticmethod
    def number_state(start):
        return ('index_state', [start])
    simplifySample = ('simplify', [])
    divide = ('divide', [])

    transitions = ('transitions', [])
