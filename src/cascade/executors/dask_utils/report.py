from bs4 import BeautifulSoup
import json
import re
import csv
import numpy as np
from dataclasses import dataclass


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
    @dataclass
    class Task:
        start: float
        duration_in_ms: float
        name: str
        worker: str
        worker_thread: str

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

        self._stream = {}
        self._task_info = {}
        for index, worker in enumerate(key_items[columns.index("worker")][1]):
            self._stream.setdefault(worker, [])
            name = key_items[name_index][1][index]
            if TaskStream.is_transfer(name):
                task_name = name
            else:
                task_name = (
                    key_items[key_index][1][index]
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                )

            self._stream[worker].append(task_name)
            self._task_info.setdefault(task_name, [])
            self._task_info[task_name].append(
                TaskStream.Task(
                    key_items[start_index][1][index],
                    key_items[duration_index][1][index],
                    task_name,
                    worker,
                    key_items[worker_thread_index][1][index],
                )
            )

    def is_transfer(task_name: str) -> bool:
        return "transfer-" in task_name

    def stream(self, exclude_transfer: bool = True) -> dict:
        if not exclude_transfer:
            return self._stream
        new_task_stream = {}
        for worker, tasks in self._stream.items():
            new_task_stream[worker] = [x for x in tasks if "transfer-" not in x]
        return new_task_stream

    def task_info(self, exclude_transer: bool = True) -> dict:
        if not exclude_transer:
            return self._task_info
        return {
            k: v for k, v in self._task_info.items() if not TaskStream.is_transfer(k)
        }


class Report:
    """
    Parses the Summary and Task Stream tabs in the Dask performance report
    """

    def __init__(self, report_file: str):
        with open(report_file) as fp:
            soup = BeautifulSoup(fp, "html.parser")
        self.summary = Summary(soup.body.script.string)
        self.task_stream = TaskStream(soup.body.script.string)


class MemoryReport:
    @dataclass
    class TaskMemory:
        min: float  # Min memory in MB
        max: float  # Max memory in MB

    def __init__(self, report_csv: str):
        self.usage = {}
        with open(report_csv) as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == "task_key":
                    continue
                self.usage.setdefault(row[0], [])
                self.usage[row[0]].append(
                    MemoryReport.TaskMemory(float(row[1]), float(row[2]))
                )
