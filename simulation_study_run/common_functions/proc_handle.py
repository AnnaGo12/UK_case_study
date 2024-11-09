import sys


def update_progress_in_command_line(percent_progess: float, time_worked_in_seconds: float):
    sys.stdout.write('\r')
    sys.stdout.write(f"progress ~ " +
                     f"{'{:.3f}'.format(percent_progess.__round__(3))}" +
                     r"%" + f"  {time_worked_in_seconds} seconds  wait...   ")
    sys.stdout.flush()


def update_progress_in_command_line_with_success(time_worked_in_seconds: float):
    sys.stdout.write('\r')
    sys.stdout.write(f"progress ~ " +
                     f"{'{:.3f}'.format(100)}" +
                     r"%" + f"  {time_worked_in_seconds} seconds  ready!   \n")
    sys.stdout.flush()
