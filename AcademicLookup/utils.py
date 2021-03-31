import re
from datetime import datetime

def get_df(title, worksheet=0, has_headers=True):
    """Returns a pandas.DataFrame representation of the
    (WORKSHEET)th worksheet of the Google Sheets (GSHEET)
    file that has title TITLE.

    TITLE - the title of the desired spreadsheet
    WORKSHEET - the index of the desired worksheet within
        the spreadsheet
    HAS_HEADERS - set to False if the spreadsheet does not
        have a header row at the top.

    It is not necessary to specify the path or the GSHEET
    file extension. Note that this creates undefined
    behavior when your google drive has multiple spreadsheets
    with the same name (i.e., you do not know which one
    will be opened).
    """
    # For details on how to handle GSHEET files, see
    # https://gspread.readthedocs.io/en/latest/api.html
    contents = gc.open(title).get_worksheet(worksheet).get_all_values()
    if has_headers:
        return pd.DataFrame.from_records(
            data=contents[1:],
            columns=contents[0]
        )
    return pd.DataFrame.from_records(contents)


def get_naturals(s):
    """Return a list of all natural numbers in the string S.

    Substrings are identified as natural numbers iff they are
    contiguous sequences of decimal digits that are not adjacent
    to other digits and that do not have leading zeros.

    >>> get_naturals('12345 abcd 12345')
    [12345, 12345]
    >>> get_naturals('3.14159')
    [3, 14159]
    >>> get_naturals('-26.4 + 0 = -26.4')
    [26, 4, 0, 26, 4]
    >>> get_naturals('012 64, 1923')
    [64, 1923]
    >>> get_naturals('00, 0--12') # If the number is just one zero, it's not a leading zero.
    [0, 12]
    """
    parts = re.split(r'[^0-9]', s)
    ret = list()
    for part in parts:
        if part and not (part[0] == '0' and part[1:]):
            ret.append(int(part))
    return ret


def get_year(s):
    """Return the numbers in S that seems most likely to be a year.

    Numbers are likely to be years if they are recent, but not later
    than the current year. (For reference, these doctests were
    written in 2021.)
    If no number is likely to be a year, then None is returned.

    >>> get_year('2019-2048.2021.apple.runningMan')
    2021
    >>> get_year('2019 20211') # 20211 is much later than the current year
    2019
    >>> get_year('01273, abc') # 102
    >>> get_year('Wolf, B, & Arnold, J. (1944). Calcium Content in ...')
    1944
    """
    ret = -1
    for n in get_naturals(s):
        if ret < n <= datetime.now().year:
            ret = n
    return ret if ret >= 0 else None


def get_extensions(filenames, min_frequency=7, max_len=5):
    """Return a list of common extensions used in the sequence
    FILENAMES.

    Extensions are only recognized if they appear at least
    MIN_FREQUENCY times and have at most MAX_LEN characters.
    """
    extensions = dict()
    for f in filenames:
        assert isinstance(f, str)
        ext = os.path.splitext(f)[1][1:]
        if ext and len(ext) <= max_len:
            extensions[ext] = extensions.get(ext, 0) + 1
    return set(
        key for key in extensions
        if extensions[key] >= min_frequency
    )


def get_words(s):
    """Returns all words and contiguous numbers in S.

    >>> get_words('Юрчак Алексей. Это было навсегда, пока не кончилось.')
    ['Юрчак', 'Алексей', 'Это', 'было', 'навсегда', 'пока', 'не', 'кончилось']
    >>> get_words('veenhof2010_ch.pdf')
    ['veenhof', '2010', 'ch', 'pdf']
    >>> get_words('Michel_1997e_Or66_lamastu.pdf')
    ['Michel', '1997', 'e', 'Or', '66', 'lamastu', 'pdf']
    >>> get_words('._transcript_KBo_IV_6_obv.htm')
    ['transcript', 'KBo', 'IV', '6', 'obv', 'htm']
    >>> get_words('macdonald_the-homeric-epics-and-the-gospel-of-mark-0300080123.pdf')
    ['macdonald', 'the', 'homeric', 'epics', 'and', 'the', 'gospel', 'of', 'mark', '0300080123', 'pdf']
    >>> get_words('Albenda, Lions Assyrian Reliefs, JANES 6, 1974b.pdf')
    ['Albenda', 'Lions', 'Assyrian', 'Reliefs', 'JANES', '6', '1974', 'b', 'pdf']
    >>> get_words('marti2009 un m%e9decin malade jmc 13.pdf')
    ['marti', '2009', 'un', 'm', 'e', '9', 'decin', 'malade', 'jmc', '13', 'pdf']
    """
    return [match.group(0) for match in re.finditer(r'([^\d\W_]+)|(\d+)', s)]


def get_search_query(s, file_extensions):
    """Constructs a search query from the file name S."""
    words = get_words(s)
    if words and words[-1] in file_extensions:
        words = words[:-1]
    return ' '.join(words)
