import enum
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Iterator, NamedTuple

from langchain_core.runnables import RunnableConfig

from langgraph.pregel import _should_interrupt
from langgraph.pregel.types import PregelExecutableTask

"""
OSS:
- memory
- zeromq
"""


class Write(NamedTuple):
    task: str
    channel: str
    value: Any


class BaseScheduler(ABC, AbstractContextManager):
    def __enter__(self):
        return self

    @abstractmethod
    def submit_tasks(self, tasks: list[PregelExecutableTask]) -> None:
        pass

    @abstractmethod
    def block_for_results(self, tasks: list[PregelExecutableTask]) -> Iterator[Write]:
        for msg in topic.listen():
            task = tasks.pop(msg.task_id)

            if task is None:
                raise

            yield Write(task, msg.channel, msg.value)

            if not tasks:
                break


class BaseCheckpointer(ABC):
    @abstractmethod
    def get_checkpoint(self) -> Any:
        pass

    @abstractmethod
    def put_checkpoint(self, checkpoint: Any) -> None:
        pass

    @abstractmethod
    def put_pending_writes(self, writes: Any) -> None:
        pass


def calc_next_tasks(checkpoint) -> list[PregelExecutableTask]:
    pass


def apply_writes(checkpoint, writes: list[Write]) -> None:
    pass


class TickStatus(enum.Enum):
    Pending = 0
    Done = 1
    InterruptBefore = 2
    InterruptAfter = 3
    OutOfSteps = 4


def tick(config: RunnableConfig, *, checkpoint=None) -> TickStatus:
    checkpoint = checkpoint or BaseCheckpointer.get_checkpoint()
    checkpoint.pending_writes

    if checkpoint.step >= config["recursion_limit"]:
        return TickStatus.OutOfSteps

    tasks = calc_next_tasks(checkpoint)

    if len({w.task for w in checkpoint.pending_writes}) != len(tasks):
        return TickStatus.Pending

    checkpoint = apply_writes(checkpoint, checkpoint.pending_writes)

    BaseCheckpointer.put_checkpoint(checkpoint)

    if _should_interrupt():
        return TickStatus.InterruptAfter

    tasks = calc_next_tasks(checkpoint)

    if not tasks:
        return TickStatus.Done

    if _should_interrupt():
        return TickStatus.InterruptBefore

    BaseScheduler.submit_tasks(tasks)

    return TickStatus.Pending


# streaming runs use below


def stream(input, config: RunnableConfig) -> Iterator[Any]:
    with BaseScheduler() as scheduler:
        while True:
            status = tick([], config, checkpoint=None)

            if status != TickStatus.Pending:
                break

            for result in scheduler.block_for_results():
                BaseCheckpointer.put_pending_writes(result.writes)
                yield result.value
