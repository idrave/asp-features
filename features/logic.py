from pathlib import Path
from typing import List
import clingo

PLASP_PATH = '/home/ivan/Documents/ai/features/plasp/plasp/build/release/bin/plasp'

class Logic:
    logicPath = Path(__file__).parent.absolute()
    #logicPath = Path('/home/ivan/Documents/ai/features/features')
    sample_file = str(logicPath/'sample/expand_states.lp')
    sample_encoding = str(logicPath/'sample/encoding.lp')
    sample_marking = str(logicPath/'mark.lp')
    grammarFile = str(logicPath/'concept.lp')
    featureFile = str(logicPath/'features.lp')
    t_g = str(logicPath/'sat_TG.lp')
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
