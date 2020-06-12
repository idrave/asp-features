import clingo
import argparse
from features.model_util import to_model_list, ModelUtil
import os

path = '/home/ivan/Documents/ai/features/features/preprocess'
clingo_file = path+'/process.lp'
def index_state(num):
    return ('index_state', [num])
def index_feature(num):
    return ('index_feature', [num])
process_sample = ('process_sample', [])
process_feature = ('process_feature', [])

def apply_program(file, program, *args):
    ctl = clingo.Control()
    ctl.load(file)
    for arg in args:
        ctl.load(arg)
    ctl.ground([('base', []), program])
    symbols = []
    with ctl.solve(yield_=True) as models:
        models = to_model_list(models)
        symbols = models[0].symbols(shown=True)
    del ctl
    return symbols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input path with sample and features')
    parser.add_argument('output', help='Output path')
    
    args = parser.parse_args()
    
    sample_out = args.output + '/index_sample.lp'
    feature_out = args.output + '/index_feature.lp'

    sample_in = args.input + '/sample'
    features_in = args.input + '/features'
    samples = None
    features = None

    if os.path.exists(sample_in) and os.path.isdir(sample_in):
        samples = os.listdir(sample_in)
        samples = [sample_in+'/'+s for s in samples]

    if os.path.exists(features_in) and os.path.isdir(features_in):
        features = os.listdir(features_in)
        features = [features_in+'/'+s for s in features]
    '''
    print('Sample')
    count = 0
    for sample in samples:
        print(sample)
        result = apply_program(clingo_file, index_state(count), sample)
        ModelUtil(result).write(sample_out, type_='a')
        count += len(result)
        del result[:]
    print(count)
    print('Features')
    
    count = 0
    for feature in features:
        print(feature)
        result = apply_program(clingo_file, index_feature(count), sample_out, feature)
        ModelUtil(result).write(feature_out, type_='a')
        count += len(result)
        del result[:]
    print(count)'''

    sample_final = args.output + '/final_sample.lp'
    feature_final = args.output + '/final_features.lp'

    for sample in samples:
        result = apply_program(clingo_file, process_sample, sample, sample_out)
        ModelUtil(result).write(sample_final, type_='a')
        del result[:]
    
    for feature in features:
        print(feature)
        result = apply_program(clingo_file, process_feature, feature, sample_out, feature_out)
        ModelUtil(result).write(feature_final, type_='a')
        del result[:]
