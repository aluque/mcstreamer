""" A python script to submit to the qsub queue. """

import sys
import os
from subprocess import call

DEF_QUEUE = 'generic'
CODE_ROOT = os.path.split(os.path.abspath(__file__))[0]


def encode_envlist(d):
    return ','.join('%s=%s' % (key, item) for key, item in d.items())

class Submission(object):
    def __init__(self, args):
        self.in_path, self.in_fname = os.path.split(os.path.realpath(args.ifile))
        
        self.rid        = os.path.splitext(self.in_fname)[0]
        self.input_file = os.path.join(self.in_path, self.in_fname)
        self.code_main  = os.path.join(CODE_ROOT, 'run.jl')
        self.log_fname  = os.path.join(self.in_path, self.rid + '.out')
        self.nthreads   = args.nthreads
        self.script     = os.path.join(CODE_ROOT, 'qrun.sh')

        self.queue      = args.queue
        self.only_print = args.only_print
        self.extra_args = args.args
        
        
    def export(self):
        env = encode_envlist({
            'INPUT_FILE': self.input_file,
            'NTHREADS': self.nthreads,
            'PROJECT_PATH': os.path.abspath(os.path.join(CODE_ROOT, os.pardir)),
            'MAIN': self.code_main})
        return env

    
    def submit_command(self):
        raise NotImplemented


    def submit(self):
        cmd = self.submit_command()
        if not self.only_print:
            call(cmd, shell=True)
        else:
            print(cmd)

            
class Qsub(Submission):
    def submit_command(self):
        args = {'-N': self.rid,
                '-j': 'oe',
                '-q': self.queue,
                '-o': 'localhost:%s' % self.log_fname,
                '-v': self.export()}

        return ('qsub %s %s %s'
                % (' '.join('%s %s' % (key, item)
                            for key, item in args.items()),
                   ' '.join(self.extra_args),
                   self.script))
           

class Slurm(Submission):
    def submit_command(self):
        args = {'--job-name': self.rid,
                '-n' : '1',
                '--cpus-per-task': self.nthreads,
                '-p': self.queue,
                '-o': '%s' % self.log_fname,
                '--export': self.export()}

        return ('sbatch %s %s %s'
                % (' '.join('%s %s' % (key, item)
                            for key, item in args.items()),
                   ' '.join(self.extra_args),
                   self.script))
           


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", "-q",
                        help="Queue to submit to", default=DEF_QUEUE)

    parser.add_argument("--system", "-s",
                        choices=["slurm", "qsub"],
                        help="Batch system to use", default='slurm')

    parser.add_argument("--nthreads", "-t", type=int,
                        help="Number of threads to use", default=1)

    parser.add_argument("--only-print", "-p",
                        action="store_true",
                        help="Just print commands, do nothing.", 
                        default=False)

    parser.add_argument("ifile", help="Input file")

    parser.add_argument("args", help="Additional args to the queuing system",
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    _class = {"slurm": Slurm,
             "qsub": Qsub}[args.system]

    
    submission = _class(args)
    submission.submit()
    

if __name__ == '__main__':
    main()
    
