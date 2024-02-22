from bs4 import BeautifulSoup
import json
import re
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
    return re.search(function_str, text).group(1)


def duration_in_sec(duration_str):
    if "ms" in duration_str:
        return float(duration_str.rstrip("ms")) / 1000
    try:
        strip_s = duration_str.rstrip("s")
        duration = float(strip_s)
    except ValueError:
        assert "m" in strip_s
        minutes, seconds = map(float, strip_s.split("m"))
        duration = 60 * minutes + seconds
    return duration


class Summary:
    def __init__(self, report_body: str):
        self.duration = duration_in_sec(search("; Duration:(.+?) &", report_body))
        self.number_of_tasks = int(search("; number of tasks:(.+?) &", report_body))
        self.compute_time = duration_in_sec(
            search("; compute time:(.+?) &", report_body)
        )
        self.transfer_time = duration_in_sec(
            search("; transfer time:(.+?) &", report_body)
        )
        self.workers = int(search("; Workers:(.+?) &", report_body))
        self.memory = search("; Memory:(.+?) &", report_body)


class TaskStream(dict):
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

        for index, worker in enumerate(key_items[columns.index("worker")][1]):
            self.setdefault(worker, [])
            name = key_items[name_index][1][index]
            if "transfer" in name:
                task_name = name
            else:
                task_name = (
                    key_items[key_index][1][index]
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                )

            self[worker].append(
                TaskStream.Task(
                    key_items[start_index][1][index],
                    key_items[duration_index][1][index],
                    task_name,
                    worker,
                    key_items[worker_thread_index][1][index],
                )
            )

    def exclude_transfer(self):
        new_task_stream = {}
        for worker, tasks in self.items():
            new_task_stream[worker] = [
                x.name for x in tasks if "transfer-" not in x.name
            ]
        return new_task_stream


class Report:
    """
    Parses the Summary and Task Stream tabs in the Dask performance report
    """

    def __init__(self, report_file: str):
        with open(report_file) as fp:
            soup = BeautifulSoup(fp, "html.parser")
        self.summary = Summary(soup.body.script.string)
        self.task_stream = TaskStream(soup.body.script.string)
