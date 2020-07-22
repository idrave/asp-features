from features.sample.sample import Sample
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
            mem.append(ps.memory_info().rss)
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


def solve_T_G_subprocess(sample: Sample, features: Features, path):
    sym = sample.get_sample() + sample.get_relevant()
    relevant_file = path+'/sample_relevant.lp'
    write_symbols(sym, relevant_file)
    cmd = ['clingo', relevant_file, features.features, Logic.t_g]
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
    return parse_clingo(out), sec, mem