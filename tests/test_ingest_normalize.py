"""Ingestion: paragraph splitting, markdown cleanup, labels, caps, quote normalization."""

from __future__ import annotations

import pytest

from hyrip.errors import DataError
from hyrip.ingest.normalize import (
    article_from_source,
    build_article,
    quote_finds,
    quote_normalized,
    split_paragraphs,
    word_count,
)
from hyrip.models.article import ArticleSource


def test_split_paragraphs_joins_hard_wrapped_lines():
    text = "First paragraph here,\nwrapped to two lines.\n\nSecond paragraph."
    paragraphs = split_paragraphs(text)
    assert paragraphs == ["First paragraph here, wrapped to two lines.", "Second paragraph."]


def test_build_article_labels_paragraphs(config, article):
    assert len(article.paragraph_labels) == len(article.paragraphs)
    assert article.paragraph_labels[:3] == ["A", "B", "C"]
    assert article.title == "Rooftop hives: the quiet boom in urban beekeeping"
    assert article.word_count == word_count(article.text)
    assert article.paragraph_labels[-1] != "A"


def test_markdown_is_stripped(config):
    import dataclasses

    config = dataclasses.replace(config, min_article_words=0)
    text = "## Heading\n\nSee [the registry](https://example.com) for **counts** of hives.\n\nDone."
    article = build_article(text, source=ArticleSource.PASTE, config=config)
    assert len(article.paragraphs) == 3
    assert article.paragraphs[0] == "Heading"
    assert article.paragraphs[1] == "See the registry for counts of hives."
    assert "#" not in article.text and "**" not in article.text and "[" not in article.text


def test_word_cap_enforced(config):
    config = config.__class__(**{**config.__dict__, "max_article_words": 10})
    with pytest.raises(DataError, match="limit is 10"):
        article_from_source(
            "one two three four five six seven eight nine ten eleven", ArticleSource.PASTE, config
        )


def test_word_floor_enforced(config):
    text = "one two three four five six seven eight nine ten eleven twelve " * 4
    with pytest.raises(DataError, match="at least"):
        article_from_source(text.strip(), ArticleSource.PASTE, config)


def test_empty_text_rejected(config):
    with pytest.raises(DataError):
        article_from_source("   \n\n  ", ArticleSource.PASTE, config)


def test_quote_normalization_handles_punctuation_and_case():
    haystack = '"People forget that a hive is a field in the sky," she said.'
    assert quote_finds("A hive is a field in the sky", haystack)
    assert quote_finds("sky, she said", haystack)
    assert not quote_finds("a hive is a barn in the sky", haystack)


def test_quote_normalized_collapses_whitespace_and_dashes():
    assert quote_normalized("state – of – the – art") == quote_normalized("state of the art")
    assert quote_normalized("Hello…  World.") == quote_normalized("hello world")


def test_curly_quotes_match_straight_ones():
    hay = "the boom’s first season"
    assert quote_finds("the boom's first season", hay)
