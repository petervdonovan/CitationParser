import re

class Name(object):
    """Stores a name."""
    def __init__(self, surname='', given_name='', 
                 given_name_initials=[]):
        """Stores the person's given name and surname."""
        assert(surname != '' and surname is not None)
        self.given_name_initials = given_name_initials
        # The first letter of every word in the given name.
        self.set_given_name(given_name)
        # The given name (anything other than the surname).
        self.surname = surname
        # The surname (also called the last name).
    def get_as_tuple(self):
        return (
            tuple(self.given_name_initials),
            self.given_name, 
            self.surname
        )
    def __hash__(self):
        """Hash is based only on surname."""
        return hash(self.get_as_tuple())
    def __eq__(self, other):
        """Returns whether this name has the same surname as another
        name.
        """
        return self.surname == other.surname
    def __lt__(self, other):
        """Returns whether this name comes alphabetically before another
        name.
        """
        return self.surname < other.surname
    def __le__(self, other):
        """Returns whether this name comes alphabetically before another
        name, or if it is equal to the other name.
        """
        return self.surname <= other.surname
    def __gt__(self, other):
        """Returns whether this name comes alphabetically after another
        name.
        """
        return self.surname > other.surname
    def __ge__(self, other):
        """Returns whether this name comes alphabetically after another
        name, or if it is equal to the other name."""
        return self.surname >= other.surname
    def contains(self, other):
        """Returns whether the other name could be the same name as this
        name. This will return true if not enough information is
        available to prove that the two names are different.
        """
        if self.surname != other.surname:
            return False
        if (self.given_name and other.given_name
                and self.given_name != other.given_name):
            return False
        if self.given_name_initials and other.given_name_initials and \
                not set(other.given_name_initials).issubset(
                    set(self.given_name_initials)):
            return False
        return True

    @staticmethod
    def get_initials_from_string(s):
        """Returns the first letter of each word.
        """
        return [match.group(0) for match in re.finditer(r'\b[a-zA-Z]', s)]
    def set_given_name(self, given_name):
        """Sets the given name and updates initials as needed."""
        self.given_name = given_name
        if given_name and not self.given_name_initials:
            # given_name is not an empty string, and initials have not
            # already been provided or computed
            self.given_name_initials = Name.get_initials_from_string(
                given_name)
    def __str__(self):
        """Returns the string representation of the name."""
        if not self.given_name:
            given_name_stand_in = ' '.join(
                initial + '.' for initial in self.given_name_initials)
        else:
            given_name_stand_in = self.given_name
        out = self.surname
        if given_name_stand_in: out += ', ' + given_name_stand_in
        return out
