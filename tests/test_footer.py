from src.config import footer_text, get_bot_name


def test_footer_exact_text_in_three_languages():
    bot = get_bot_name()
    expected = {
        "en": f"{bot} is AI and can make mistakes.",
        "fr": f"{bot} est une IA et peut faire des erreurs.",
        "ar": f"{bot} ذكاء اصطناعي وقد يرتكب أخطاء.",
    }
    for language, sentence in expected.items():
        text = footer_text(language)
        assert text == sentence
        assert bot in text


def test_footer_contains_no_external_link_or_html_link():
    for language in ("en", "fr", "ar"):
        text = footer_text(language).lower()
        assert "http://" not in text
        assert "https://" not in text
        assert "www." not in text
        assert "<a" not in text
        assert "href" not in text
        assert "]" not in text  # no markdown link syntax
