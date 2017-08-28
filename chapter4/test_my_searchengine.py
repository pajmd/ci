from chapter4.my_searchengine import *
import sqlite3

import mock

@mock.patch.object(Crawler, 'addtoindex')
@mock.patch.object(Crawler, 'isindexed')
@mock.patch.object(Crawler, 'addlinkref')
def _test_crawler(mock_addtoindex, mock_isindexed, mock_addlinkref):
    mock_addtoindex.return_value = None
    mock_isindexed.return_value = False
    mock_addlinkref.return_value = None
    crawler = Crawler('crawler.db')
    pages = [
        'https://en.wikipedia.org/wiki/Decision_tree',
        'http://scikit-learn.org/stable/modules/tree.html'
    ]
    crawler.crawl(pages)


def _test_crawler_db():
    crawler = Crawler('crawler.db')
    try:
        crawler.createindextables()
    except sqlite3.OperationalError as e:
        print('DB already exists')

    # 'http://scikit-learn.org/stable/modules/tree.html'
    pages = [
        'https://en.wikipedia.org/wiki/Decision_tree',
        'https://www.toptal.com/python/an-introduction-to-mocking-in-python'
    ]
    crawler.crawl(pages, 1)

def test_query():
    searcher = Searcher('crawler.db')
#    searcher.getmatchrows('decision tree learning')
    # searcher.getmatchrows('RapidMiner stuff python')
    # searcher.getmatchrows('decision mock python')
    # searcher.getmatchrows('decision tree python')
    searcher.getmatchrows('true false')

def test_count():
    from collections import Counter
    l = [('a',1), ('b', 'stuff'), ('a', 'stuffy'),('b',3),('b',4)]
    c = Counter(t[0] for t in l)
    print(c['a'], c['b'])
    l='si j etait une prhase'
    c = Counter(l)
    print(c)
    d = dict(c)
    print(d)

def test_frequecy_score():
    rows = [
        (1, 44, 29),
        (1, 44, 46),
        (1, 44, 87),
        (1, 44, 142),
        (1, 44, 144),
        (1, 44, 149),
        (191, 3087, 3906),
        (191, 3087, 3989),
        (191, 3087, 5149),
        (191, 3087, 5669),
        (191, 3087, 5389)
    ]
    searcher = Searcher('crawler.db')
    d1 = searcher.frequencyscore(rows)
    d2 = searcher.frequencyscore_better(rows)
    assert len(d1.items() & d2.items()) == len(d1) == len(d2)
