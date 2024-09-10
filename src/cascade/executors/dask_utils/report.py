import json
import re

import numpy as np
from bs4 import BeautifulSoup


def find_key_values(key, dic):
    res = None
    for search_key, search_items in dic.items():
        if search_key == key:
            res = search_items
        if isinstance(search_items, dict):
            res = find_key_values(key, search_items)
        elif isinstance(search_items, list):
            for item in search_items:
                if isinstance(item, dict):
                    res = find_key_values(key, item)
                    if res is not None:
                        break
        if res is not None:
            return res
    return None


def search(function_str, text):
    res = re.search(function_str, text)
    if res is not None:
        res = res.group(1)
    return res


def duration_in_sec(duration_str):
    remaining = duration_str
    times = []
    for unit in ["hr", "m", "s", "ms", "us"]:
        if unit in remaining:
            time, next = remaining.split(unit)
            if not time.endswith(("m", "u")) and not next.startswith("s"):
                times.append(float(time))
                remaining = next
                continue
        times.append(0)
    return np.multiply(times, [60 * 60, 60, 1, 1 / 1000, 1 / 1000000]).sum()


class Summary:
    def __init__(self, report_body: str):
        self.duration = duration_in_sec(search("; Duration:(.+?) &", report_body))
        self.number_of_tasks = int(search("; number of tasks:(.+?) &", report_body))
        self.compute_time = duration_in_sec(
            search("; compute time:(.+?) &", report_body)
        )
        transfer = search("; transfer time:(.+?) &", report_body)
        if transfer is not None:
            self.transfer_time = duration_in_sec(transfer)
        else:
            self.transfer_time = 0
        self.workers = int(search("; Workers:(.+?) &", report_body))
        self.memory = search("; Memory:(.+?) &", report_body)


class TaskStream:
    class Task:
        def __init__(
            self,
            start: float,
            duration: float,
            name: str,
            worker: str,
            worker_thread: str,
        ):
            self.start = start
            self.duration_in_ms = duration
            self.end = start + duration
            self.name = name
            self.worker = worker
            self.worker_thread = worker_thread

        def is_transfer(self) -> bool:
            return "transfer-" in self.name

    def __init__(self, report_body: str):
        report_dict = json.loads(report_body)
        task_stream = report_dict[list(report_dict.keys())[0]]["roots"][0][
            "attributes"
        ]["tabs"][1]["attributes"]
        key_items = find_key_values("entries", task_stream)
        columns = [item[0] for item in key_items]
        start_index = columns.index("start")
        name_index = columns.index("name")
        key_index = columns.index("key")
        duration_index = columns.index("duration")
        worker_thread_index = columns.index("worker_thread")

        self._stream: dict[str, list["TaskStream.Task"]] = {}
        for index, worker in enumerate(key_items[columns.index("worker")][1]):
            self._stream.setdefault(worker, [])
            name = key_items[name_index][1][index]
            if "transfer-" in name:
                task_name = name
            else:
                task_name = (
                    key_items[key_index][1][index]
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                )

            self._stream[worker].append(
                TaskStream.Task(
                    key_items[start_index][1][index],
                    key_items[duration_index][1][index],
                    task_name,
                    worker,
                    key_items[worker_thread_index][1][index],
                )
            )
            # Order tasks by start time
            self._stream[worker].sort(key=lambda x: x.start)
        self.start = min(
            [x.start for task_stats in self._stream.values() for x in task_stats]
        )
        self.end = max(
            [x.end for task_stats in self._stream.values() for x in task_stats]
        )

    def wall_time(self) -> float:
        """
        Total wall time of task stream in seconds
        """
        return (self.end - self.start) / 1000

    def stream(
        self, exclude_transfer: bool = True
    ) -> dict[str, list["TaskStream.Task"]]:
        if not exclude_transfer:
            return self._stream
        new_task_stream = {}
        for worker, tasks in self._stream.items():
            new_task_stream[worker] = [x for x in tasks if not x.is_transfer()]
        return new_task_stream

    def task_info(
        self, exclude_transfer: bool = True
    ) -> dict[str, list["TaskStream.Task"]]:
        tasks: dict[str, list["TaskStream.Task"]] = {}
        for task_stats in self._stream.values():
            for task in task_stats:
                if exclude_transfer and task.is_transfer():
                    continue
                tasks.setdefault(task.name, []).append(task)
        return tasks

    def is_enclosed(self, worker: str, task: "TaskStream.Task") -> bool:
        start_index = self._stream[worker].index(task)
        end_times = [x.end for x in self._stream[worker]]
        end_times.sort()
        end_index = end_times.index(task.end)
        if end_index < start_index:
            return True
        return False

    def idle_time(self, percentage: bool = False) -> dict[str, float]:
        """
        Idle time for each worker, either as a absolute value in seconds or as a
        percentage of total time. Idle time is time when worker is not executing tasks
        or transferring data. Note, extracting idle times from Dask Performance Report
        does not give times matching those displayed in the Dask dashboard.

        Params
        ------
        percentage: bool, where to return percentages of total time or absolute values in
        seconds

        Returns
        -------
        dict[str, float]: A dictionary of worker idle times with keys as worker names
        """
        worker_stats = {}
        for worker, tasks in self._stream.items():
            idle = 0.0
            for index, current_info in enumerate(tasks):

                if index == 0:
                    idle += current_info.start - self.start
                else:
                    if self.is_enclosed(worker, current_info):
                        continue
                    # Need to account for transfer tasks over lapping with and enclosing
                    # compute tasks
                    previous_index = index - 1
                    while self.is_enclosed(worker, tasks[previous_index]):
                        previous_index -= 1
                    previous_info = tasks[previous_index]
                    if previous_info.end < current_info.start:
                        idle += current_info.start - previous_info.end

            # Add on idle time after last completing task
            idle += self.end - max(x.end for x in self._stream[worker])
            # Convert to seconds
            idle /= 1000
            worker_stats[worker] = idle * 100 / self.wall_time() if percentage else idle
        return worker_stats

    def transfer_time(
        self, percentage: bool = False, blocking: bool = True
    ) -> dict[str, float]:
        """
        Time spent on data transfer for each worker, either as a absolute value in seconds or as a
        percentage of total time. For each worker, the total time spent in blocking data transfer
        is returned if blocking is True.

        Params
        ------
        percentage: bool, where to return percentages of total time or absolute values in seconds
        blocking: bool, where to return only blocking transfer time. If False, time spent on all data
        transfer tasks is returned.

        Returns
        -------
        dict[str, float]: A dictionary of worker transfer times with keys as worker names
        """
        worker_stats = {}
        for worker, tasks in self._stream.items():
            blocking_time = 0.0
            total_time = 0.0
            for index, task in enumerate(tasks):
                if task.is_transfer():
                    if self.is_enclosed(worker, task):
                        total_time += task.duration_in_ms
                        continue

                    # Check overlap with previous and next tasks
                    previous_index = index - 1
                    next_index = index + 1
                    previous = tasks[previous_index] if previous_index >= 0 else None
                    next = tasks[next_index] if next_index < len(tasks) else None

                    blocking_time += task.duration_in_ms
                    total_time += task.duration_in_ms
                    if previous is not None and previous.end > task.start:
                        blocking_time -= previous.end - task.start
                    if next is not None and next.start < task.end:
                        blocking_time -= task.end - next.start
            transfer = blocking_time if blocking else total_time
            # Convert to seconds
            transfer /= 1000
            worker_stats[worker] = (
                transfer * 100 / self.wall_time() if percentage else transfer
            )
        return worker_stats

    def stats(self, percentage: bool = True) -> tuple[float, float]:
        """
        Statistics on time (seconds) or percentage of time spent across all workers being idle or on
        blocking data transfer.

        Params
        ------
        percentage: bool, where to return percentages of total time or absolute values in seconds

        Returns
        -------
        tuple[float, float]: A tuple of idle time and blocking data transfer time
        """
        idle_times = self.idle_time(False)
        transfer_times = self.transfer_time(percentage=False, blocking=True)
        idle = sum(idle_times.values())
        transfer = sum(transfer_times.values())

        total_time = self.wall_time() * len(self._stream)
        if percentage:
            return idle * 100 / total_time, transfer * 100 / total_time
        return idle, transfer


class Report:
    """
    Parses the Summary and Task Stream tabs in the Dask performance report
    """

    def __init__(self, report_file: str):
        with open(report_file) as fp:
            soup = BeautifulSoup(fp, "html.parser")
        self.summary = Summary(soup.body.script.string)
        self.task_stream = TaskStream(soup.body.script.string)
