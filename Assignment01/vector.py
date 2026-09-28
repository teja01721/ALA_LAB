import math


class Vector:
    def __init__(self, entries):
        self._entries = list(entries)

    def mean(self):
        """Return the mean of the vector entries."""
        return sum(self._entries) / len(self._entries)

    def demean(self):
        """Return a new vector with the mean subtracted from each entry."""
        mean = self.mean()
        return Vector([x - mean for x in self._entries])

    def std(self):
        """Return the standard deviation of the vector entries."""
        demeaned = self.demean()
        squared_deviations = [x ** 2 for x in demeaned._entries]

        return math.sqrt(sum(squared_deviations) / len(self._entries))
