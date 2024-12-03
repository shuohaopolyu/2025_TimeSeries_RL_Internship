def is_subset(a, b):
    return set(a).issubset(set(b))

def power_set(lst):
    if len(lst) == 0:
        return [[]]
    return power_set(lst[1:]) + [[lst[0]] + x for x in power_set(lst[1:])]