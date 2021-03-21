import re
from RegexBased.Citations.citationRegexes import styleGuideRegexes, \
                                      nameListRegexes, nameRegexes, \
                                      getParts
from RegexBased.Utils.regexes import regexes
from People.Name import Name
import datetime

class Citation(object):
    """Store and process string references (sources cited in articles)"""
    countYearAmbiguous = 0
    countYearNotGiven = 0
    def __init__(self, raw):
        '''Stores basic information about the citation, from which other
        information will be derived.'''
        self.raw = raw
        '''The raw string of this citation.'''
    def __str__(self):
        '''Returns the string representation of this Citation.'''
        return self.raw
    def getStyleGuides(self):
        '''Returns a list of possible style guides for this citation.'''
        styleGuides = []
        for styleGuide in styleGuideRegexes:
            if re.match(styleGuideRegexes[styleGuide], self.raw):
                styleGuides.append(styleGuide)
        return styleGuides
    def _getNameListMatch(self):
        '''Returns the name list format description and Match object
        corresponding to the contributor names from the citation, if
        such a list exists.'''
        matches = []
        for nameListType in nameListRegexes:
            match = re.match(nameListRegexes[nameListType], self.raw)
            if match:
                matches.append((nameListType, match))
        longestMatch = ('', None)
        for match in matches:
            if not longestMatch[1] or len(match[1].group(0)) > len(longestMatch[1].group(0)):
                longestMatch = match
        return longestMatch
    def getNameList(self):
        '''Returns the list of contributor names, as it appears in the
        original raw text.
        '''
        nameList = self._getNameListMatch()[1]
        if nameList is None:
            return None
        return nameList.group(0)
    def getNames(self):
        '''Returns a list of contributor names that appear in the 
        citation, in order of appearance in the citation.'''
        # Get namelist
        nameList = self._getNameListMatch()
        # get the upper-level nameregexes that appear in the namelist
        if not nameList[0]: return []
        names = []
        nameRegexesInNameList = [
            nameRegex for nameRegex in nameRegexes 
            if nameRegexes[nameRegex] in nameListRegexes[nameList[0]]
            ]
        try:
            nameRegexesInNameList.remove('name')
        except ValueError: pass
        try:
            nameRegexesInNameList.remove('initial')
        except ValueError: pass
        # get matches with those regexes
        nl = nameList[1].group(0)
        while len(nl) > 1:
            for regex in nameRegexesInNameList:
                match = re.match(nameRegexes[regex], nl)
                if match:
                    names = [name for name in names if name[1] not in match.group(0)]
                    if not any(match.group(0) in name[1] for name in names):
                        names.append((regex, match.group(0)))
                    nl = ' ' + nl.replace(match.group(0), '')
            nl = nl[1:]
        # Get namepartses
        namePartses = [getParts(name[0], name[1]) for name in names]
        # convert namepartses into Names and return
        return [
            Name(
                surname=parts['surname'], 
                givenName=parts['givenName']
                    + ' '.join(parts['middleName']),
                givenNameInitials=parts['givenNameInitials']
                    + parts['middleNameInitials']
                )
            for parts in namePartses
            if parts['surname']
            ]
    def getYear(self):
        # contained by parentheses
        years = getCitableYearsFromString(r'([^\da-zA-Z]\(', r'[a-z]\)[^\da-zA-Z])', self.raw)
        if not years:
            # in its own sentence
            years = getCitableYearsFromString(r'([^a-z,] ', r'\.(( )|$))', self.raw)
        if not years:
            # in its own word
            years = getCitableYearsFromString(r'(\b', r'\b)', self.raw)
        if not years:
            # at least not combined with another number
            years = getCitableYearsFromString(r'(([^0-9]|^)', r'([^0-9]|$))', self.raw)
        if years and max(years) != years[0] or (
            bool(getCitableYearsFromString(r'([^\da-zA-Z]\(', r'[a-z]\)[^\da-zA-Z])', self.raw)) and 
            bool(getCitableYearsFromString(r'([^a-z] ', r'\.(( )|$))', self.raw))
            ):
            Citation.countYearAmbiguous += 1
        try:
            return years[0]
        except IndexError:
            Citation.countYearNotGiven += 1
            return None
    def getTitle(self):
        """Returns a substring that is likely to be the title of the
        work.
        """
        text = self.raw
        nameList = self.getNameList()
        if nameList:
            endFirstWord = text.find(' ') % len(text)
            nameListStart = text.find(nameList)
            if nameListStart < endFirstWord:
                text = text[nameListStart + len(nameList.strip()):]
                text = text[text.find(' '):].strip()
        year = str(self.getYear())
        if year:
            endFirstWord = text.find(' ') % len(text)
            yearStart = text.find(year)
            if yearStart < endFirstWord:
                text = text[endFirstWord:].strip()
        match = (re.match(regexes['title in quotes'], text)
            or  re.match(regexes['title in single quotes'], text)
            or  re.match(regexes['title case sentence'], text)
            or  re.match(regexes['sentence'], text))
        return match.group(0) if match else None


def getCitableYearsFromString(frontRe, backRe, string):
    return [
        int(re.search(r'\d+', result[0]).group()) 
        for result in re.findall(
            frontRe + regexes['citable year'] + backRe, string
            )
        if int(re.search(r'\d+', result[0]).group())  <= int(datetime.datetime.now().year)
        ]
