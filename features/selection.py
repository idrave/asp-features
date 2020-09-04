from features.sample.sample import SampleView, Sample
from features.feat import Features
import features.solver as solver
from features.solver import SolverType, ClingoProfiling
from features.logic import Logic
from features.model_util import write_symbols
import subprocess
import psutil
import time
import multiprocessing
import clingo
import re
import argparse
import sys
from pathlib import Path

def solve_T_G(sample: Sample, features: Features):
    with solver.create_solver(type_= SolverType.PROFILE) as ctl:
        ctl.load([Logic.t_g, features.features])
        ctl.addSymbols(sample.get_sample() + sample.get_relevant())
        ctl.ground([Logic.base])
        prof_g = ctl.get_profiling()
        sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        prof_s = ctl.get_profiling()
    return sym, prof_g, prof_s

def profile(pipe, pid):
    ps = psutil.Process(pid=pid)
    SLEEP_TIME=0.00025
    mem = []
    while True:
        if ps.is_running():
            try:
                mem.append(ps.memory_info().rss)
            except:
                print('Process finished') #TODO
        if pipe.poll(timeout=SLEEP_TIME):
            pipe.recv()
            break
    pipe.send(mem)
    pipe.close()

def parse_clingo(output):
    regex = re.compile(r'Answer: \d+?.*?\n(.*?\n)Optimization: (\d+?)')
    parsed = regex.findall(output)
    selected = re.compile(r'selected\(([^,]*?)\)$')
    result = []
    for line in parsed:
        sym_str = line[0].split('\n')[0].split()
        cost = int(line[1])
        sym = []
        for s_str in sym_str:
            s = selected.match(s_str)
            assert(s != None)
            sym.append(clingo.Function('selected', [int(s[1])]))
        result.append((sym, cost,))

    return result

def solve_T_G_subprocess_dir(sample_file, features_file, threads=1):
    cmd = ['clingo', sample_file, features_file, Logic.t_g, '-t', str(threads)]
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    prof = multiprocessing.Process(target=profile, args=(child_conn, proc.pid))
    prof.start()
    child_conn.close()
    out = proc.stdout.read().decode('utf-8')
    proc.wait()
    sec = time.time() - start
    parent_conn.send(1)
    mem = parent_conn.recv()
    parent_conn.close()
    prof.join()
    return out, sec, mem

def solve_T_G_subprocess(sample: SampleView, path, threads=1):
    sym = sample.get_states() + sample.get_transitions() + sample.symbols.get_atoms('goal', 1) + sample.get_relevant()
    relevant_file = path+'/sample_relevant.lp'
    features_file = str(Path(path)/'features.lp')
    write_symbols(sym, relevant_file)
    out, sec, mem = solve_T_G_subprocess_dir(relevant_file, features_file, threads=threads)
    with open(path+'/clingo_stdout.txt','w') as fp:
        fp.write(out)
    return parse_clingo(out), sec, mem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    parser.add_argument('features')
    parser.add_argument('output')
    parser.add_argument('-t', '--threads')
    args = parser.parse_args()
    out, t, mem = solve_T_G_subprocess_dir(args.sample, args.features, threads=args.threads)
    solution = parse_clingo(out)
    message = \
        'Profiling samples: {}\n'.format(len(mem)) + \
        'Solving took {}s, start memory {} MB, max memory {} MB\n'.format(round(t, 3), round(min(mem)/1e6, 3), round(max(mem)/1e6, 3)) + \
        'Solutions found: {}\n'.format(len(solution)) + \
        'Optimal solution: {}. Cost: {}.\n'.format(*solution[-1] if len(solution) > 0 else (None, None))
    print(message)
    with open(args.output,'w') as fp:
        fp.write('Call to: ' + str(sys.argv) + '\n')
        fp.write(message+'\n')
        fp.write(out)