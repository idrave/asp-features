import clingo
from model_util import ModelUtil, to_model_list

relevant_feature = [("bool", 1), ("num", 1), ("feature", 1), ("transition", 2), ("relevant", 2), ("goal", 1), ("cost", 2), ("hasValue", 3)]

def features(concept_file, sample_file):
    path = '/home/ivan/Documents/ai/features/features'
    ctl = clingo.Control()
    ctl.load(concept_file)
    ctl.load(sample_file)
    ctl.load(path+"/features.lp")
    ctl.ground([("base", []), ("features", [])])
    print("Starting to solve")
    with ctl.solve(yield_=True) as models:
        models = to_model_list(models)
        features = ModelUtil(models[0].symbols(atoms=True)).get_symbols(filter=relevant_feature)
    return features

if __name__ == "__main__":
    import sys
    concepts = sys.argv[1]
    samples = sys.argv[2]
    output = sys.argv[3]
    result = features(concepts, samples)
    ModelUtil(result).write(output)
    