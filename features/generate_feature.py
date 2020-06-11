import clingo
from model_util import ModelUtil, to_model_list


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
        features = models[0].symbols(shown=True)
    return features

if __name__ == "__main__":
    import sys
    concepts = sys.argv[1]
    samples = sys.argv[2]
    output = sys.argv[3]
    result = features(concepts, samples)
    ModelUtil(result).write(output)
    