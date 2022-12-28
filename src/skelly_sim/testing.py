import contextlib
import os
import pytest

from pathlib import Path
from subprocess import run

@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

@pytest.fixture(scope="session")
def sim_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("sims")
    return path

def run_sim(path: Path=Path('.')):
    print("Running simulation")
    with working_directory(path):
        res = run(['skelly_sim', '--overwrite'], shell=False, capture_output=False)
        return not res.returncode

def run_precompute(path: Path=Path('.')):
    print("Generating precompute data")
    with working_directory(path):
        res = run(['skelly_precompute'], shell=False, capture_output=True)
        return not res.returncode
