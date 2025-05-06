from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import json
from functools import partial
def process_multi_thread(data_l, process, *args):
    final_data = []
    process_fn = lambda data : process(data, *args) if args else process
    with ThreadPoolExecutor(max_workers=(multiprocessing.cpu_count())) as execute:
        futures = {execute.submit(process_fn, data): data for data in data_l}
        for i, future in enumerate(as_completed(futures)):
            try:
                resp_result = future.result()
                final_data.append(resp_result)
            except Exception as e:
                print(f"Error processing future {i}: {e}")
    return final_data