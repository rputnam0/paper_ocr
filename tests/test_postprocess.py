from paper_ocr.postprocess import parse_yaml_front_matter


def test_parse_yaml_front_matter():
    text = """---\nfoo: bar\nnum: 2\n---\nHello"""
    parsed = parse_yaml_front_matter(text)
    assert parsed.metadata["foo"] == "bar"
    assert parsed.metadata["num"] == 2
    assert parsed.markdown == "Hello"


def test_no_front_matter():
    text = "Hello"
    parsed = parse_yaml_front_matter(text)
    assert parsed.metadata == {}
    assert parsed.markdown == "Hello"
