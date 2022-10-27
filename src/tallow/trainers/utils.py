from .trainer import EarlyStopHook, Trainer, TrainerStateDict


def retrieve_best_model(trainer: Trainer) -> TrainerStateDict:
    for hook in trainer.hook_mgr:
        if isinstance(hook, EarlyStopHook):
            return hook._disk_mgr.load()
    raise ValueError()
