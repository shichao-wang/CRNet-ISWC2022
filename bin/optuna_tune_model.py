import os
import re
from asyncio.log import logger
from typing import Callable, List, Mapping

import optuna
from molurus import hierdict
from train_model import train

OptunaTuneFunction = Callable[[str, hierdict.HierDict], float]


class OptunaTuner:
    def __init__(
        self,
        save_folder_path: str,
        train_model: OptunaTuneFunction,
        base_cfg: hierdict.HierDict,
        hpspace: Mapping,
        studyname: str = "optuna",
        direction: str = "maximize",
    ) -> None:
        os.makedirs(save_folder_path, exist_ok=True)
        self.save_folder_path = save_folder_path
        self.train_model = train_model
        self.base_cfg = base_cfg
        self.hpspace = hpspace

        storage = f"sqlite:///{self.save_folder_path}/{studyname}.db"
        self.study = optuna.create_study(
            storage=storage,
            study_name=studyname,
            load_if_exists=True,
            direction=direction,
        )

    def execute(
        self, num_trials: int = None, num_seconds: int = None
    ) -> optuna.Trial:
        try:
            self.study.optimize(self._objective, num_trials, num_seconds)
        except KeyboardInterrupt:
            pass
        best_trial = self.study.best_trial
        return best_trial

    def _objective(self, trial: optuna.Trial):
        trial_cfg = suggest_cfg(self.base_cfg, trial, self.hpspace)
        trial_folder_path = os.path.join(
            self.save_folder_path, str(trial.number)
        )
        return self.train_model(trial_folder_path, trial_cfg)


def suggest_cfg(
    base_cfg: hierdict.HierDict, trial: optuna.Trial, hp_space: Mapping
):
    cfg = base_cfg.copy()
    for key in hp_space:
        cfg[key] = trial.suggest_categorical(key, hp_space[key])
    return cfg


TIME_RE = re.compile(r"^(?P<num>\d+)(?P<unit>[dhm]$)")

UNIT_SECONDS = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60}


def parse_timestr(time_string: str):
    matches = TIME_RE.match(time_string)
    if matches is None:
        raise ValueError()
    number, unit = matches.group("num", "unit")
    return int(float(number) * UNIT_SECONDS[unit])


def main():
    parser = hierdict.ArgumentParser()
    parser.add_argument("--save-folder-path")
    parser.add_argument("--num-trials", default=None, required=False)
    parser.add_argument("--run-time", default=None, required=False)
    args = parser.parse_args()
    base_cfg = hierdict.parse_args(args.cfg, args.overrides)
    hpspace = base_cfg.pop("hpspace")

    num_seconds = None
    if args.run_time is not None:
        num_seconds = parse_timestr(args.run_time)
        logger.info("Run for %s (%d seconds)", args.run_time, num_seconds)

    tuner = OptunaTuner(args.save_folder_path, train, base_cfg, hpspace)
    best_trial = tuner.execute(args.num_trials, num_seconds)
    print(best_trial)


if __name__ == "__main__":
    main()
