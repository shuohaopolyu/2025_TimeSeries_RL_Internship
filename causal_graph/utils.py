def is_subset(a, b):
    return set(a).issubset(set(b))

def power_set(lst):
    if len(lst) == 0:
        return [[]]
    return power_set(lst[1:]) + [[lst[0]] + x for x in power_set(lst[1:])]

def check_consistency(full_output_vars: list[str]) -> bool:
    var_names = []
    for output_var in full_output_vars:
        var_name, time_step = output_var.split("_")
        if int(time_step) != full_output_vars.index(output_var):
            return False
        var_names.append(var_name)
    return len(set(var_names)) == 1

def check_element_index(full_do_vars) -> bool:
    # not tested
    for do_vars in full_do_vars:
        for do_var in do_vars:
            var_name, time_step = do_var.split("_")
            if int(time_step) != full_do_vars.index(do_vars):
                return False
    return True
        