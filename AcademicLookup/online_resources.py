from bs4 import BeautifulSoup
from utils import get_year
import requests
import scholarly
    
class Search:
    """Encapsulates the results of a search for metadata."""
    def result(self):
        """Returns a summary of the results of the query."""
        return {
            attr: method()
            for attr, method in [
                ('authors', self.authors),
                ('language', self.language),
                ('publication_year', self.publication_year)
            ]
        }

class AutoSearch(Search):
    """Encapsulates the results of a search of the query itself."""
    def __init__(self, query, min_year=1900):
        self.query = query
        self.min_year = min_year
    def publication_year(self):
        y = get_year(self.query)
        if y and y >= self.min_year:
            return y
    def authors(self):
        return None # This is not implemented
    def language(self):
        return None # This is not implemented

class WorldCatSearch(Search):
    """Encapsulates the results of a WorldCat search."""
    WORLDCAT = 'https://www.worldcat.org'

    def __init__(self, query, min_year=1900):
        """Makes a WorldCat query with the keywords specified in QUERY
        and saves the result in SELF.
        """
        self.results = BeautifulSoup(
            requests.get(WorldCatSearch.WORLDCAT + '/search', {
                'q': query,
                'qt': 'results_page'
            }).content,
            'lxml'
        ).select('tr.menuElem')
        self.min_year = min_year

    def authors(self):
        """Returns the author(s) associated with the first result of the
        query.
        Returns None if the desired information is unavailable.
        """
        try:
            text = self.results[0].select('.result.details .author')[0].text
        except IndexError:
            return None
        if text[:3] == 'by ':
            return text[3:]
        return text

    def language(self):
        """Returns the language associated with the first result of the
        query. Returns None if the desired information is unavailable.
        """
        try:
            return self.results[0].select('.result.details .itemLanguage'
                )[0].text
        except IndexError:
            return None
    
    def publication_year(self):
        try:
            pub_data = self.results[0].select('.publisher')[0].strings
        except IndexError:
            return None
        for s in pub_data:
            y = get_year(s)
            if y and y >= self.min_year:
                return y
        return None

    def cover_art(self):
        """Returns a link to the cover art associated with the first result
        of the query.
        Returns None if the desired information is unavailable.
        """
        try:
            return self.results[0].select('.coverart img')[0]['src']
        except IndexError:
            return None

    def worldcat_link(self):
        """Returns a link to a WorldCat page with details about the
        queried bibliographic resource.
        """
        try:
            return WorldCatSearch.WORLDCAT + \
                self.results[0].select('.result.details .name a')[0]['href']
        except IndexError:
            return None


class GoogleBooksSearch(Search):
    """Encapsulates the results of a Google Books search."""
    GOOGLE_BOOKS = 'https://www.googleapis.com/books/v1/volumes'

    def __init__(self, query, min_year=None):
        """Makes a Google Books query with the keywords specified in
        QUERY and saves the result in SELF.
        """
        self.response = requests.get(
            'https://www.googleapis.com/books/v1/volumes', {
                'q': query
            }).json()

    def _get_volume_info(self, key):
        if self._first_result():
            try:
                return self._first_result()['volumeInfo'][key]
            except KeyError:
                return None

    def authors(self):
        """Returns the author(s) associated with the first result of the
        query, separated by semicolons.
        """
        authors = self._get_volume_info('authors')
        return '; '.join(authors) if authors else None

    def language(self):
        """Returns the language associated with the first result of the
        query.
        """
        return self._get_volume_info('language')

    def publication_year(self):
        y = self._get_volume_info('publishedDate')
        return get_year(y) if isinstance(y, str) else y
    
    def google_books_link(self):
        """Returns a link to a Google Books page with details about the
        queried bibliographic resource.
        Returns None if the desired information is unavailable.
        """
        if self._first_result():
            return self._first_result()['selfLink']
    
    def _first_result(self):
        """Returns the first result of the query."""
        try:
            return self.response['items'][0]
        except KeyError:
            return None

class GoogleScholarSearch(Search):
    """Encapsulates the results of a Google Scholar search."""
    def __init__(self, query, min_year=1900):
        try:
            self.response = scholarly.search_single_pub(query)
        except IndexError:
            self.response = None
        self.min_year = min_year
    def authors(self):
        if self.response:
            return '; '.join(self.response['bib'])
    def language(self):
        return None # This is not implemented
    def publication_year(self):
        if self.response:
            y = get_year(self.response['bib']['pub_year'])
            return y if y >= self.min_year else None

class CompoundSearch(Search):
    def __init__(
            self,
            query,
            min_year=1900,
            search=[AutoSearch, WorldCatSearch, GoogleBooksSearch],
            verbose=False):
        self.responses = [
            (s.__name__, s(query, min_year).result())
            for s in search
        ]
        self.verbose = verbose
    def _choose_data(self, key):
        """Returns a value corresponding to KEY from the list of
        possible values given by the searches that have been attempted.
        """
        answers = []
        for search, result in self.responses:
            answer = result[key]
            if answer:
                answers.append(answer)
                if self.verbose:
                    print('{} says {}={}.'.format(search, key, answer))
        def n_appearances(value):
            return sum(1 if ans == value else 0 for ans in answers)
        if answers:
            ret = max(answers, key=n_appearances)
            return ret
        return None
    def publication_year(self):
        return self._choose_data('publication_year')
    def language(self):
        return self._choose_data('language')
    def authors(self):
        return self._choose_data('authors')