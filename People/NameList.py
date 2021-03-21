from People.Name import Name
import warnings

class NameList:
    """Represents a list of names, including source text the data it
    communicates. Note that a certain degree of stability in the
    implementation of this data type is in order because it is intended
    to be pickled.
    """
    def __init__(self, raw, names):
        """Initializes a new NameList object."""
        assert isinstance(raw, str)
        assert all(isinstance(name, Name) for name in names)
        self.raw = raw
        self.names = names
    def surnames(self):
        """Returns a list of the surnames represented in SELF."""
        return [name.surname for name in self.names]
    def given_names(self):
        """Returns a list of the given names represented in SELF."""
        return [name.given_name for name in self.names]
    def contains(self, name):
        """Returns whether NAME may be represented in SELF."""
        return any(self_name.contains(name) for self_name in self.names)
    @classmethod
    def delimited(cls, raw, sep=';', minor_sep=','):
        """Creates a new NameList from RAW, a string in which author
        names are separated from each other by SEP and surnames
        are separated from given names by MINOR_SEP (with surnames
        preceding given names).
        """
        if raw is None:
            return None
        names = list()
        for name_raw in raw.split(sep):
            parts = name_raw.strip().split(minor_sep)
            if len(parts) > 2:
                warnings.warn('Warning: One author\'s name apparently '
                      'is represented with multiple internal separators.'
                      ' Meaning is unclear.')
            elif len(parts) < 2:
                warnings.warn('Warning: One author\'s name apparently '
                      'is not represented with any internal separators.'
                      ' Meaning is unclear.')
            names.append(Name(
                surname=parts[0],
                given_name=minor_sep.join(parts[1:]).strip()
            ))
        return cls(raw, names)