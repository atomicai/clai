import logging
import os
import pathlib
import time
from typing import Optional

from icecream import ic

from clai.logging import base

logger = logging.getLogger(__name__)


class ICELogger(base.BaseLogger):

    save_dir = str(pathlib.Path(os.getcwd()) / ".logging")
    offset_step = 0
    sync_step = True

    def _time_stamp(self):
        return '%i |> ' % int(time.time())

    @classmethod
    def init_experiment(
        cls,
        experiment_name,
        project_name="icecream",
        api: Optional[str] = None,
        notes=None,
        tags=None,
        entity=None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        _id: Optional[str] = None,
        log_checkpoint: Optional[bool] = False,
        sync_step: Optional[bool] = True,
        prefix: Optional[str] = None,
        notebook: Optional[str] = None,
        **kwargs,
    ):

        prefix = cls._time_stamp() if prefix is None else prefix
        save_dir = cls.save_dir if save_dir is None else str(pathlib.Path(save_dir))
        ic.configureOutput(prefix=prefix)

        return cls(tracking_uri=None)

    def log_metrics(self, metrics, step, **kwargs):
        assert self.experiment is not None, "Initialize experiment first by calling `WANDBLogger.init_experiment(...)`"
        metrics = {f"{self.prefix}{k}": v for k, v in metrics.items()}
        if self.sync_step and step + self.offset_step < self.experiment.step:
            logger.warning("Trying to log at a previous step. Use `sync_step=False`")
        if self.sync_step:
            self.experiment.log(metrics, step=(step + self.offset_step) if step is not None else None)
        elif step is not None:
            self.experiment.log({**metrics, 'step': step + self.offset_step}, **kwargs)
        else:
            self.experiment.log(metrics)
