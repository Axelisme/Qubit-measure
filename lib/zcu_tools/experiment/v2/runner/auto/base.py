from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from ..base import AbsTask, BatchTask, ResultType, TaskContext


class AbsAutoTask(AbsTask):
    def __init__(
        self,
        needed_tags: List[str] = [],
        optional_tags: List[str] = [],
        provided_tags: List[str] = [],
    ) -> None:
        self.needed_tags = needed_tags
        self.optional_tags = optional_tags
        self.provided_tags = provided_tags

    @abstractmethod
    def run(
        self, ctx: TaskContext, need_infos: Dict[str, complex]
    ) -> Dict[str, complex]:
        """Run the task with current context and needed information. Return provided information."""


class AutoBatchTask(BatchTask):
    def __init__(self, tasks: Dict[str, AbsAutoTask]) -> None:
        if any(name == "meta_infos" for name in tasks.keys()):
            raise ValueError("'meta_infos' is a reserved name for the meta information")

        self.tasks = tasks

        self.task_pbar = None

    def run(self, ctx: TaskContext) -> None:
        assert self.task_pbar is not None
        self.task_pbar.reset()

        meta_infos: Dict[str, complex] = {}
        task_sequence = self.generate_task_sequence(self.tasks)

        for name, task in task_sequence:
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")

            # collect needed information from previous tasks
            need_infos: Dict[str, complex] = {}
            for name, task in self.tasks.items():
                for tag in task.provided_tags:
                    if tag in task.needed_tags or tag in task.optional_tags:
                        # only last matching task's information is used
                        need_infos[tag] = meta_infos[name][tag]

            task.init(ctx, keep=False)
            with ctx(addr=name):
                provided_infos = task.run(ctx, need_infos)
            task.cleanup()

            meta_infos[name] = provided_infos
            self.task_pbar.update()

        # set meta information to context
        with ctx(addr="meta_infos"):
            ctx.set_data(meta_infos)

    def get_default_result(self) -> ResultType:
        default_result = {
            name: task.get_default_result() for name, task in self.tasks.items()
        }
        default_result["meta_infos"] = {
            name: {tag: np.array(np.nan, dtype=complex) for tag in task.provided_tags}
            for name, task in self.tasks.items()
        }
        return default_result

    @staticmethod
    def generate_task_sequence(
        tasks: Dict[str, AbsAutoTask],
    ) -> List[Tuple[str, AbsAutoTask]]:
        """
        Generate a valid task execution sequence based on dependencies.

        This method performs a topological sort of the tasks, respecting the
        `needed_tags` as hard dependencies. It also attempts to optimize the
        sequence to satisfy as many `optional_tags` as possible by using a
        greedy approach. At each step, it selects the next task that has all
        its hard dependencies met and satisfies the maximum number of optional
        dependencies based on the tags provided by already sequenced tasks.

        Args:
            tasks: A dictionary of task names to task objects.

        Returns:
            A list of (task_name, task_object) tuples in a valid execution order.

        Raises:
            ValueError: If a dependency cannot be met (e.g., a needed_tag is
                        never provided, or a circular dependency is detected).
        """
        # 1. Map tags to the tasks that provide them
        provided_tags_map: Dict[str, List[str]] = {}
        for name, task in tasks.items():
            for tag in task.provided_tags:
                provided_tags_map.setdefault(tag, []).append(name)

        # 2. Build the dependency graph (adjacency list and in-degrees)
        adj: Dict[str, List[str]] = {name: [] for name in tasks}
        in_degree: Dict[str, int] = {name: 0 for name in tasks}

        # First, determine the unique set of prerequisite tasks for each task
        prerequisites: Dict[str, set[str]] = {name: set() for name in tasks}
        for name, task in tasks.items():
            for needed_tag in task.needed_tags:
                providers = provided_tags_map.get(needed_tag)
                if not providers:
                    raise ValueError(
                        f"Task '{name}' needs tag '{needed_tag}', which is not provided by any task."
                    )
                for provider_name in providers:
                    prerequisites[name].add(provider_name)

        # Now, build the graph from the unique prerequisites
        for name, deps in prerequisites.items():
            in_degree[name] = len(deps)
            for provider_name in deps:
                adj[provider_name].append(name)

        # 3. Initialize the queue with tasks that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]

        sorted_sequence: List[str] = []
        available_tags: set[str] = set()

        # 4. Process tasks using a modified Kahn's algorithm
        while queue:
            # Prioritize tasks that satisfy more optional dependencies
            scored_tasks = []
            for task_name in queue:
                task = tasks[task_name]
                satisfied_optionals = available_tags.intersection(task.optional_tags)
                score = len(satisfied_optionals)
                # Sort by score (desc), then by name (asc) for deterministic tie-breaking
                scored_tasks.append((-score, task_name))

            scored_tasks.sort()

            if not scored_tasks:
                # Should not happen if queue is not empty, but as a safeguard
                break

            best_task_name = scored_tasks[0][1]

            # Move the selected task from the queue to the sorted sequence
            u = best_task_name
            queue.remove(u)
            sorted_sequence.append(u)

            # Update the set of available tags
            available_tags.update(tasks[u].provided_tags)

            # Update the in-degrees of dependent tasks
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # 5. Check for circular dependencies
        if len(sorted_sequence) != len(tasks):
            remaining_tasks = set(tasks.keys()) - set(sorted_sequence)
            raise ValueError(
                f"A circular dependency was detected among tasks: {remaining_tasks}"
            )

        return [(name, tasks[name]) for name in sorted_sequence]
