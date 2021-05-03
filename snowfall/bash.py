from pathlib import Path
from typing import Optional
import subprocess

from tqdm.auto import tqdm


class ParallelBash:
    def __init__(self):
        self.processes = []
        self.logfiles = []

    def run(self, cmd: str, log_path: Optional[Path] = None):
        self.logfiles.append(log_path.open('w') if log_path is not None else None)
        self.processes.append(
            subprocess.Popen(
                cmd,
                shell=True,
                text=True,
                stdout=self.logfiles[-1],
                stderr=self.logfiles[-1]
            )
        )

    def join(self, msg: Optional[str] = None, progress_bar: bool = True):
        for i, (p, logf) in enumerate(tqdm(zip(self.processes, self.logfiles), desc=msg, disable=not progress_bar)):
            rc = p.wait()
            if rc != 0:
                print(f'Error when executing task {i}' + (
                    f': see {logf.name} for details.'
                    if logf is not None
                    else '.'
                )
                      )
            if logf is not None:
                logf.close()
        self._reset()

    def _reset(self):
        self.processes = []
        self.logfiles = []

