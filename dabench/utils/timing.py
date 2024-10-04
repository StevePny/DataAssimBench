import datetime
import time


def report_timing(timing_label=""):

    if not hasattr(report_timing, "timing_start_time"):
        report_timing.timing_start_process_time = time.process_time()
        report_timing.timing_start_time = time.time()
        report_timing.last_process_time = report_timing.timing_start_process_time
        report_timing.last_time = report_timing.timing_start_time
        return

    print(f"\n< === {timing_label} ===")

    # Print the current time, helpful for tracking long runs
    now = datetime.datetime.now()
    print(f"Current datetime is: {now}")

    print("  ===  ")

    # get process execution time
    timing_end_process_time = time.process_time()
    seconds = timing_end_process_time - report_timing.last_process_time
    minutes = seconds / 60.0
    print(f"CPU Execution time of this step: {seconds} seconds or {minutes} minutes.")
    seconds = timing_end_process_time - report_timing.timing_start_process_time
    minutes = seconds / 60.0
    print(f"CPU Execution time so far: {seconds} seconds or {minutes} minutes.")

    print("  ===  ")

    # get wall clock time
    timing_end_time = time.time()
    seconds = timing_end_time - report_timing.last_time
    minutes = seconds / 60.0
    print(
        f"Wall Clock Execution time of this step: {seconds} seconds or {minutes} minutes."
    )
    seconds = timing_end_time - report_timing.timing_start_time
    minutes = seconds / 60.0
    print(f"Wall Clock Execution time so far: {seconds} seconds or {minutes} minutes.")

    print(f"  === {timing_label} === >\n")

    # Set up to get estimate of time between calls
    report_timing.last_process_time = timing_end_process_time
    report_timing.last_time = timing_end_time


def _test():
    report_timing(timing_label="initializing...")
    time.sleep(3)
    report_timing(timing_label="3 second sleep.")
    time.sleep(10)
    report_timing(timing_label="10 second sleep.")


# %% Main access
if __name__ == "__main__":
    # main(sys.argv)
    _test()
