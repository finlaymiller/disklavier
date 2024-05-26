from multiprocessing import Process, cpu_count
import redis
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)


def process_json_keys(redis_url, start_index, end_index, subr_index=0):
    """Process JSON keys in Redis using indices to determine the range of keys each process handles.

    Args:
        redis_conn (Redis): Redis connection object.
        start_index (int): The index of the first key this process will handle.
        end_index (int): The index of the last key this process will handle.
    """
    print(f"[SUBR{subr_index:02d}] starting scan from {start_index} - {end_index}")
    r = redis.Redis.from_url(redis_url)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    update_task = progress.add_task(
        f"[SUBR{subr_index:02d}] updating objects", total=end_index - start_index
    )

    with progress:
        cursor = "0"
        current_index = 0
        while True:
            print(f"[SUBR{subr_index:02d}] cursor is at {cursor}")
            cursor, keys = r.scan(cursor=cursor, count=10000)  # type: ignore
            if current_index + len(keys) < start_index:
                # Skip these keys as they are before the start_index
                current_index += len(keys)
                continue

            for key in keys:
                if start_index <= current_index < end_index:
                    key_type = r.execute_command("TYPE", key)
                    if key_type == b"ReJSON-RL":
                        value = r.json().get(key)

                        if value:
                            row_file, col_file, metric = str(key).split(":")

                            value["row_file"] = row_file[2:]
                            value["col_file"] = col_file
                            value["metric"] = metric

                            r.json().set(key, "$", value)

                current_index += 1
                if current_index >= end_index:
                    return subr_index

            if cursor == "0":
                break
            progress.advance(update_task)


def distribute_processing(redis_url):
    """Distribute Redis JSON processing across available CPU cores by evenly dividing the keys among them.

    Args:
        redis_url (str): URL for the Redis connection.
    """
    r = redis.Redis.from_url(redis_url)
    total_keys = r.dbsize()
    num_processes = cpu_count()
    keys_per_process = total_keys // num_processes  # type: ignore
    extra_keys = total_keys % num_processes  # type: ignore

    processes = []
    start_index = 0
    for i in range(num_processes):
        # Distribute the extra keys among the first few processes
        end_index = start_index + keys_per_process + (1 if i < extra_keys else 0)
        process = Process(
            target=process_json_keys, args=(redis_url, start_index, end_index, i)
        )
        processes.append(process)
        process.start()
        start_index = end_index

    # Wait for all processes to complete
    for process in processes:
        subr_i = process.join()
        print(f"[MAIN]   subroutine {subr_i} complete")


if __name__ == "__main__":
    distribute_processing("redis://localhost:6379")
