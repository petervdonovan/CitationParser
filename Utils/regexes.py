'''
Any constant regular expressions that are better represented as 
a variable name than as a raw string literal should be contained in this document.
'''

regexes = {
    'long month': r'((January)|(February)|(March)|(April)|(May)|(June)|(July)|(August)|(September)|(October)|(November)|(December))',
    'short month': r'((Jan\.?)|(Feb\.?)|(Mar\.?)|(Apr\.?)|(May\.?)|(June?\.?)|(July?\.?)|(Aug\.?)|(Sept\.?)|(Oct\.?)|(Nov\.?)|(Dec\.?))',
    #'3-letter month no period':r'((Jan)|(Feb)|(Mar)|(Apr)|(May)|(Jun)|(Jul)|(Aug)|(Sep)|(Oct)|(Nov)|(Dec))',
    'season': '((Spring)|(Summer)|(Fall)|(Autumn)|(Winter)|(spring)|(summer)|(fall)|(autumn)|(winter))',
    'recent year': r'20[0-2][0-9]', #in the 2000s
    'citable year': r'(((14)|(15)|(16)|(17)|(18)|(19)|(20))[0-9]{2})', #Between 1400 and present
    'name': r'[^\b0-9\.,a-z\(\)‘"\'’“”—\s][^\s0-9\.,?!\"”]+',
    'initial': '[^\s0-9a-z\.,\(\)‘"\'’“”\`~!@#$%^&\*\(\)_\+=\-<>\?/;:—]',
    'title in quotes': r'["“][^.]*[\.\?!]["”]',
    'title in single quotes': r'[‘\'][^.]*[’\']',
    #'url': r'((http)|(www))', #TODO: DEFINE WHAT MAKES A LINK.
    'url': r'[^ ]*?',
    'day of month': r'([1-3]?[0-9])',
    'title case sentence': r'([A-Z][A-Za-z]+)(( [A-Z][A-Za-z]+)|( [a-z]{1,3})|[ :\"“”\'\(\)\d])*[\.\?\!]',
    'sentence': r'[^\.\?\!]+[\.\?\!]'
    }
'''Dictionary containing Regex patterns of some common words/phrases.'''
