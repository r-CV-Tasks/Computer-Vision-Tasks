
#
# Helpers Functions Implementations
#

def map_ranges(inputValue: float, inMin: float, inMax: float, outMin: float, outMax: float):
    """
    Map a given value from range 1 -> range 2
    :param inputValue: The value you want to map
    :param inMin: Minimum Value of Range 1
    :param inMax: Maximum Value of Range 1
    :param outMin: Minimum Value of Range 2
    :param outMax: Maximum Value of Range 2
    :return: The new Value in Range 2
    """
    slope = (outMax - outMin) / (inMax - inMin)
    return outMin + slope * (inputValue - inMin)