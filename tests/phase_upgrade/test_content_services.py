"""V3.upgrade — content service parser smoke tests (no network)."""
from __future__ import annotations

from services.news_service import NewsService


_SAMPLE_RSS = '''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Top stories</title>
<item>
<title>Story One</title>
<link>https://example.com/1</link>
<pubDate>Wed, 07 May 2026 21:00:00 GMT</pubDate>
<source>Reuters</source>
</item>
<item>
<title>Story Two</title>
<link>https://example.com/2</link>
<pubDate>Wed, 07 May 2026 22:00:00 GMT</pubDate>
<source>AP</source>
</item>
<item>
<title></title>
<link>https://example.com/3</link>
</item>
</channel></rss>'''


def test_news_rss_parser_extracts_titled_items():
    headlines = NewsService._parse_rss(_SAMPLE_RSS, max_results=10)
    assert len(headlines) == 2  # the empty-title item is dropped
    assert headlines[0].title == 'Story One'
    assert headlines[0].source == 'Reuters'
    assert headlines[1].title == 'Story Two'


def test_news_rss_parser_respects_max_results():
    headlines = NewsService._parse_rss(_SAMPLE_RSS, max_results=1)
    assert len(headlines) == 1


def test_news_rss_parser_handles_garbage():
    assert NewsService._parse_rss('not xml', max_results=5) == []
    assert NewsService._parse_rss('<html><body>not rss</body></html>', max_results=5) == []
