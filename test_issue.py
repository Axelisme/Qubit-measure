
from typing import Callable, Sequence, Any, List
from dataclasses import dataclass

@dataclass
class TaskContext:
    cfg: Any
    data: Any

class AbsTask:
    pass

class HardTask(AbsTask):
    def __init__(self, measure_fn):
        self.measure_fn = measure_fn

class SoftTask(AbsTask):
    def __init__(
        self,
        sweep_name: str,
        sweep_values: Sequence[Any],
        update_cfg_fn: Callable[[int, TaskContext, Any], None],
        sub_task: AbsTask,
    ) -> None:
        self.sweep_values = sweep_values
        self.sweep_name = sweep_name
        self.update_cfg_fn = update_cfg_fn
        self.sub_task = sub_task

class Runner:
    def __init__(self, task: AbsTask, update_hook=None):
        self.task = task

def test():
    times = [1, 2, 3]
    measure_fn = lambda ctx, hook: None
    
    try:
        task = SoftTask(
            sweep_name="times",
            sweep_values=times,
            update_cfg_fn=lambda _, ctx, time: None,
            sub_task=HardTask(measure_fn=measure_fn),
        )
        runner = Runner(
            task=task,
            update_hook=lambda ctx: None
        )
        print("Success")
    except TypeError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()

