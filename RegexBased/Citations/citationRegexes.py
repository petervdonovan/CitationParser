from Utils.regexes import regexes
from collections import OrderedDict
import re
def getParts(namesRegex, str):
        '''Returns a dictionary containing the parts of a name, gotten 
        from a raw string and the namesRegex that describes the raw string.'''
        surname = ''
        givenName = ''
        middleName = ''
        initials = []
        givenNameInitials = []
        middleNameInitials = []
        nr = nameRegexes['name']
        ir = nameRegexes['initial']
        if namesRegex == 'surname, given name':
            names = [groups for groups in re.findall(nr, str)]
            surname = names[0]
            givenName = names[1]
        elif (namesRegex == 'surname, given name middle name' or 
            namesRegex == 'surname, given name and optionally middle name'):
            surname = getParts('surname, given name', str)['surname']
            givenName = getParts('surname, given name', str)['givenName']
            middleName = [
                groups for groups in re.findall(nr, str) 
                if groups[0] not in surname and 
                   groups[0] not in givenName
                ]
            if middleName: middleName = middleName[0]
            else: middleName = ''
        elif namesRegex == 'surname, given name and optionally middle initial':
            surname = getParts('surname, given name', str)['surname']
            givenName = getParts('surname, given name', str)['givenName']
            str2 = str.replace(surname, '').replace(givenName, '')
            middleNameInitials = [groups[0] for groups in re.findall(ir, str2)]
        elif namesRegex == 'given name optionally middle name surname':
            names = [groups for groups in re.findall(nr, str)]
            givenName = names[0]
            surname = names[-1]
            if len(names) > 2:
                middleName = ' '.join(names[1:-1])
        elif namesRegex == 'given name optionally middle initial surname':
            surname = getParts('given name optionally middle name surname', str)['surname']
            givenName = getParts('given name optionally middle name surname', str)['givenName']
            str2 = str.replace(surname, '').replace(givenName, '')
            middleNameInitials = [groups[0] for groups in re.findall(ir, str2)]
        elif namesRegex == 'name in order':
            if re.match(nameRegexes['given name optionally middle name surname'], str):
                return getParts('given name optionally middle name surname', str)
            else:
                return getParts('given name optionally middle initial surname', str)
        elif (namesRegex in [
            'surname, first initial', 
            'surname, initials', 
            'surname, initials with dot optional space optional', 
            'surname, initials with dot', 
            'surname, initials with dot space optional', 
            'surname initial(s)', 
            'surname, initial(s) no dots no spaces'
            ]):
            surname = re.findall(nr, str)[0]
            str2 = str.replace(surname, '')
            initials = [groups[0] for groups in re.findall(ir, str2)]
        elif (namesRegex in ['initials with dot space surname', 'initials with dot surname']):
            surname = re.findall(nr, str)[-1]
            str2 = str.replace(surname, '')
            initials = [groups[0] for groups in re.findall(ir, str2)]
        if initials:
            if not givenNameInitials:
                givenNameInitials = [initials[0]]
                if len(initials) > 1:
                    middleNameInitials = initials[1:]
            else:
                middleNameInitials = initials
        return {
            'surname': surname,
            'givenName': givenName,
            'middleName': middleName,
            'givenNameInitials': givenNameInitials,
            'middleNameInitials': middleNameInitials
            }
# Names regexes
nameRegexes = {
    'name': r'[^\b0-9\.,a-z\(\)‘"\'’“”—\s][^\s0-9\.,?!\"”]+',
    'initial': '[^\s0-9a-z\.,\(\)‘"\'’“”\`~!@#$%^&\*\(\)_\+=\-<>\?/;:—]',
    }
nameRegexes['surname, given name'] = r'(' + nameRegexes['name'] + r', ' + nameRegexes['name'] + r')'
nameRegexes['surname, given name middle name'] = r'(' + nameRegexes['surname, given name'] + r' ' + nameRegexes['name'] + r')'
nameRegexes['surname, given name and optionally middle name'] = r'(' + nameRegexes['surname, given name'] + r'( ' + nameRegexes['name'] + r')?)'
nameRegexes['surname, given name and optionally middle initial'] = r'(' + nameRegexes['surname, given name'] + r' ' + nameRegexes['initial'] + '\.( ' + nameRegexes['initial'] + r'\b){0,2})'
nameRegexes['given name optionally middle name surname'] = r'(' + nameRegexes['name'] + r' ' + nameRegexes['name'] + r' (' + nameRegexes['name'] + r')?)'
nameRegexes['given name optionally middle initial surname'] = r'(' + nameRegexes['name'] + r' (' + nameRegexes['initial'] + '\. )?' + nameRegexes['name'] + r')'
nameRegexes['name in order'] = r'(' + nameRegexes['given name optionally middle name surname'] + r'|' + nameRegexes['given name optionally middle initial surname'] + r')'
nameRegexes['surname, first initial'] = r'(' + nameRegexes['name'] + r', ' + nameRegexes['initial'] + '\.)'
nameRegexes['surname, initials'] = r'(' + nameRegexes['name'] + r' ' + nameRegexes['initial'] + '{1,3}\b)'
nameRegexes['surname, initials with dot optional space optional'] = r'(' + nameRegexes['name'] + r',? (\.? ?' + nameRegexes['initial'] + '){1,3}((\.)|\b))'
nameRegexes['surname, initials with dot'] = r'(' + nameRegexes['name'] + r',?( ' + nameRegexes['initial'] + '\.){1,3})'
nameRegexes['surname, initials with dot space optional'] = r'(' + nameRegexes['name'] + r',? ( ?' + nameRegexes['initial'] + '\.){1,3})'
nameRegexes['surname initial(s)'] = r'(' + nameRegexes['name'] + r' ' + nameRegexes['initial'] + '(' + nameRegexes['initial'] + '){0,2}\b)'
nameRegexes['surname, initial(s) no dots no spaces'] = r'(' + nameRegexes['name'] + r',? [^\s0-9a-z\.,‘"\'’“”\(\)\[\]]([^\s0-9a-z\.,‘"\'’“”\(\)\[\]]){0,2}\b)'
nameRegexes['initials with dot space surname'] = r'((' + nameRegexes['initial'] + '\. ){0,3}' + nameRegexes['name'] + r')'
nameRegexes['initials with dot surname'] = r'((' + nameRegexes['initial'] + '\. ?){0,3} ' + nameRegexes['name'] + r')'
# Name lists
nameListRegexes = OrderedDict()
nameListRegexes['ama name list'] = r'(' + nameRegexes['surname initial(s)'] + r'(, ' + nameRegexes['surname initial(s)'] + r')*?((\.)|(, et al.)))'
nameListRegexes['apa name list'] = r'(' + nameRegexes['surname, initials with dot optional space optional'] + r'((, ' + nameRegexes['surname, initials with dot optional space optional'] + r')*' + \
        r',? (&|((\.){3,4})|…) ' + nameRegexes['surname, initials with dot optional space optional'] + r')?)'
nameListRegexes['chicago/turabian name list'] = r'(' + r'(((' + nameRegexes['surname, given name and optionally middle name'] + r')|(' + \
    nameRegexes['surname, given name and optionally middle initial'] + r'))((, (' + nameRegexes['given name optionally middle name surname'] + \
    r'|' + nameRegexes['given name optionally middle initial surname'] + r'))*?,? ((and)|e|y) ' + r'(' + \
    nameRegexes['given name optionally middle name surname'] + r'|' + nameRegexes['given name optionally middle initial surname'] + r')' + r')?((\.)|,)' + r')' \
    + r'|' + nameRegexes['surname, given name and optionally middle initial'] + r')'
nameListRegexes['harvard: australian name list'] = r'(' + nameRegexes['surname, initial(s) no dots no spaces'] + r'((, ' + nameRegexes['surname, initial(s) no dots no spaces'] + r')*' + \
    r' & ' + nameRegexes['surname, initial(s) no dots no spaces'] + r')?)'

nameListRegexes['mla name list'] = r'((' + nameRegexes['surname, given name and optionally middle name'] + r'((, et al)?)|' + \
        nameRegexes['surname, given name and optionally middle initial'] + r'(, et al)?)' + \
        r'(, (((and)|e|y) )?' + nameRegexes['given name optionally middle initial surname'] + r')?' + r'\.)'
nameListRegexes['harvard name list'] = r'(' + nameRegexes['surname, initials with dot'] + r'(, ' + \
         nameRegexes['surname, initials with dot'] + r')*( ((and)|e|y) ' + nameRegexes['surname, initials with dot'] + \
         r')?( et al.)?' + r')'
#nameListRegexes['chicago/turabian name list no end on dot requirement'] = r'(' + r'(((' + nameRegexes['surname, given name and optionally middle name'] + r')|(' + \
#    nameRegexes['surname, given name and optionally middle initial'] + r'))((, (' + nameRegexes['given name optionally middle name surname'] + \
#    r'|' + nameRegexes['given name optionally middle initial surname'] + r'))*?, ((and)|e|y) ' + r'(' + \
#    nameRegexes['given name optionally middle name surname'] + r'|' + nameRegexes['given name optionally middle initial surname'] + r')' + r')?\.?' + r')' \
#    + r'|' + nameRegexes['surname, given name and optionally middle initial'] + r')'
#Non-style-guide-specific nameLists
nameListRegexes['with semicolons'] = r'(' + nameRegexes['surname, initials with dot space optional'] + '(; ' + nameRegexes['surname, initials with dot space optional'] + r')*(, ((and)|e|y) ' + nameRegexes['surname, initials with dot space optional'] + r')?)'
nameListRegexes['pure surname initials'] = r'((' + nameRegexes['surname, initials with dot optional space optional'] + r',? )+)'
nameListRegexes['pure surname initials no dots no spaces'] = r'((' + nameRegexes['surname, initial(s) no dots no spaces'] + r'(,|(\.))? (and )?)+(et al.)?)'
nameListRegexes['pure initials surname with dots'] = r'((' + nameRegexes['initials with dot surname'] + r'(,|(\.))?( ((and)|e|y))? )+(et al.)?)'
nameListRegexes['surname initials then initials surname'] = r'(' + nameRegexes['surname, initials with dot'] + r'(, ' + nameRegexes['initials with dot space surname'] + r')*' + r'(,? ((and)|e|y) ' + nameRegexes['initials with dot space surname'] + r')?)'
nameListRegexes['surname initials et al.'] = r'(' + nameRegexes['surname, initials with dot space optional'] + r', et al\.' + r')'
nameListRegexes['surname initials list with "and"'] = r'(' + nameRegexes['surname, initials with dot space optional'] + r'((, ' + nameRegexes['surname, initials with dot space optional'] + r')*' + \
        r',? ((and)|e|y) ' + nameRegexes['surname, initials with dot space optional'] + r')?)'
nameListRegexes['apa variant, given name first'] = r'(' + nameRegexes['surname, given name and optionally middle name'] + r'((, ' + nameRegexes['name in order'] + r')*' + \
        r',? (&|((\.){3,4})|…) ' + nameRegexes['name in order'] + r')?[^\sa-z])'
#nameListRegexes['pure surname initials list'] = r'(' + nameRegexes['surname, initials with dot space optional'] + r'(, ' + nameRegexes['surname, initials with dot space optional'] + r')*)'
'''
The following style guides were found in the EBSCO citation generator.
Initial development set for regexes for apa, mla, and chicago:
- the first 5 results for a search for 'lopez' on JSTOR
- the first 5 results for a search for 'robert' on JSTOR
This development set was used both to initially create the regexes and to initially "test" them.
Subsequent development set:
The first 5 results for the following searches on EBSCO academic search complete (after finding that stopwords yield no results in their search)
- drowning
- dessication
- blue
'''
styleGuideRegexes = {
    'abnt': r'random',
    'ama': r'(' + nameListRegexes['ama name list'] + r' [^"“].*?' + regexes['recent year'] + r';.*$)',
    #'apa': r'(^' + nameListRegexes['apa name list'] + r') \(' + regexes['recent year'] + r'\)\. ' + r'.*?[^\.]$',
    'apa': r'(^' + nameListRegexes['apa name list'] + r') \(' + regexes['recent year'] + r'\)\. ' + r'.*?$',
    'chicago': r'(^' + nameRegexes['surname, given name and optionally middle name'] + r'\. )?' \
        + regexes['title in quotes'] + r' In .*?Accessed ' + regexes['long month'] + r' ' + \
        regexes['day of month'] + r', ' + regexes['recent year'] + r'\. [^\s]*?\.$', #does not exist in EBSCO citation generator, but does exist in JSTOR
    'chicago/turabian: author-date': r'(' + nameListRegexes['chicago/turabian name list'] + r' ' + \
        regexes['recent year'] + r'\. ' + regexes['title in quotes'] + r'.*?\.$)',
    'harvard: australian': r'(' + nameListRegexes['harvard: australian name list'] + r' ' + \
        regexes['recent year'] + r', ' + regexes['title in single quotes'] + r'.*?viewed ' + \
        regexes['day of month'] + r' ' + regexes['long month'] + r' ' + regexes['recent year'] + \
        r', \<' + regexes['url'] + r'\>\.$)',
    'harvard': r'(' + nameListRegexes['harvard name list'] + r' \(' + regexes['recent year'] + r'\) ' + \
        regexes['title in single quotes'] + r'.*?\.$)', 
    'chicago/turabian: humanities': r'(' + nameListRegexes['chicago/turabian name list'] + r' ' + \
        regexes['title in quotes'] + r'.*?(\(((' + regexes['season'] + r')|(' + regexes['long month'] + \
        r')|(' + regexes['short month'] + r')) (' + regexes['day of month'] + r', )?' + \
        regexes['recent year'] + r'\)\:).*?\.$)',
    #'mla': r'^(((((' + nameRegexes['surname, given name and optionally middle name'] + r'|' + nameRegexes['surname, given name and optionally middle initial'] + r'), (and )?)' + \
    #    r'(' + nameRegexes['given name optionally middle name surname'] + r'\.|' + nameRegexes['given name optionally middle initial surname'] + r'))|(' + 
    #    nameRegexes['surname, given name and optionally middle name'] + r'\.|' + nameRegexes['surname, given name and optionally middle initial'] + r')) )?' + \
    #    regexes['title in quotes'] + ' .*?(Accessed ' + regexes['day of month'] + ' ' + regexes['short month'] + \
    #    ' ' + regexes['recent year'] + ')?\.$',
    'mla': r'^(' + nameListRegexes['mla name list'] + r' )?' + regexes['title in quotes'] + \
        r'.*?' + regexes['recent year'] + r', .*?\.$',
    'vancouver/icmje': r'(' + nameListRegexes['ama name list'] + r'.*?' + regexes['recent year'] + \
        r' (' + regexes['short month'] + r'|' + regexes['season'] + r').*?Available from:.*?$)'
    }
'''Dictionary containing Regex patterns of some of the common style guides.'''
