"""Corrected entry point. See revised/PROTOCOL.md before reproducing."""
import os,argparse
from threadpoolctl import threadpool_limits
from revised.data import prepare
from revised.experiments import run
from revised.final import final
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 os.environ['SWEETTV_DATA']=a.data;os.environ['SWEETTV_REVISION_OUT']=a.output
 with threadpool_limits(limits=2):
  prepare(a.data,a.output);run('main',a.output);run('forward',a.output);final(a.output)
