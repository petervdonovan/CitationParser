import re

class Name(object):
    """Stores a name."""
    def __init__(self, surname='', givenName='', 
                 givenNameInitials=[]):
        """Stores the person's given name and surname."""
        assert(surname != '' and surname is not None)
        self.givenNameInitials = givenNameInitials
        # The first letter of every word in the given name.
        self.setGivenName(givenName.lower().capitalize())
        # The given name (anything other than the surname).
        self.surname = surname.lower().capitalize()
        # The surname (also called the last name).
    def getAsTuple(self):
        return (
            tuple(self.givenNameInitials),
            self.givenName, 
            self.surname
        )
    def __hash__(self):
        """Hash is based only on surname."""
        return hash(self.getAsTuple())
    def __eq__(self, other):
        """Returns whether this name has the same surname as another name."""
        return self.surname == other.surname
    def __lt__(self, other):
        """Returns whether this name comes alphabetically before another name."""
        return self.surname < other.surname
    def __le__(self, other):
        """Returns whether this name comes alphabetically before another name,
        or if it is equal to the other name."""
        return self.surname <= other.surname
    def __gt__(self, other):
        """Returns whether this name comes alphabetically after another name."""
        return self.surname > other.surname
    def __ge__(self, other):
        """Returns whether this name comes alphabetically after another name,
        or if it is equal to the other name."""
        return self.surname >= other.surname
    def contains(self, other):
        """Returns whether the other name could be the same name as this name.
        This will return true if not enough information is available to prove 
        that the two names are different."""
        if self.surname != other.surname:
            return False
        if self.givenName and other.givenName and self.givenName != other.givenName:
            return False
        if self.givenNameInitials and other.givenNameInitials and \
            not set(other.givenNameInitials).issubset(set(self.givenNameInitials)):
            return False
        return True

    @staticmethod
    def getInitialsFromString(str):
        """Returns the first letter of any substring involving the first
        letter of a word and then a period.
        """
        return [match.group(0)[0] for match in re.findall(r'\b\.', str)]
    def setGivenName(self, givenName):
        """Sets the given name and updates initials as needed."""
        self.givenName = givenName
        if givenName: # givenName is not an empty string
            givenNameInitials = Name.getInitialsFromString(givenName)
    def __str__(self):
        """Returns the string representation of the name."""
        if not self.givenName:
            givenNameStandIn = ' '.join(initial + '.' for initial in self.givenNameInitials)
        else:
            givenNameStandIn = self.givenName
        out = self.surname
        if givenNameStandIn: out += ', ' + givenNameStandIn
        return out
