import time
from collections import OrderedDict

# Static Variable decorator
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


@static_var('timers', OrderedDict())
def timer(name, print_one=False, print_all=False, clean_all=False, output_file=None):
    """
    Timer to profile the code. It measures the time elapsed between successive
    calls and it prints the total time elapsed after sucesive calls.
    """
    if name is None:
        pass
    elif name not in timer.timers:
        timer.timers[name] = (0, time.time())
    elif timer.timers[name][1] is None:
        time_tuple = (timer.timers[name][0], time.time())
        timer.timers[name] = time_tuple
    else:
        time_tuple = (timer.timers[name][0] + (time.time() - timer.timers[name][1]), None)
        timer.timers[name] = time_tuple
        if print_one is True:
            print(name, ' = ', timer.timers[name][0])

    if print_all is True:
        col_width = max(len(key) for key in timer.timers)
        for key in timer.timers:
            print("".join(key.ljust(col_width)), ' = ', timer.timers[key][0])
        if output_file is not None:
            with open(output_file, 'w') as f:
                for key in sorted(timer.timers):
                    f.write("".join(key.ljust(col_width)) + ' = ' + str(timer.timers[key][0]) + '\n')

    if clean_all:
        timer.timers = {}
    return
