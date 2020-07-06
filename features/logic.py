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


class Feature:
    processFeat = ('feature', [])
    comparePreFeature = ('compare_prefeature', [])
    compareFeature = ('compare_feature', [])
    pruneFeature = ('prune_feature', [])
    @staticmethod
    def numberFeat(start):
        return ('number_feat', [start])
    @staticmethod
    def divide_feat(start, first, gsize):
        return ('divide_feat', [start, first, gsize])
    classifyFeat = ('classify_feat', [])
    primitiveFeature = ('primitiveFeature', [])
    conceptFeature = ('conceptFeature', [])
    @staticmethod
    def distFeature(k):
        return ('distFeature', [k])
