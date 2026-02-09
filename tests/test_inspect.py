from paper_ocr.inspect import compute_text_heuristics, decide_route, is_text_only_candidate


def test_decide_route_anchor():
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"text": "Hello world " * 60}]},
                ],
            }
        ]
    }
    h = compute_text_heuristics(page_dict)
    assert decide_route(h, mode="auto") == "anchored"
    assert is_text_only_candidate(h)


def test_decide_route_unanchored():
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"text": "cid:123 cid:456"}]},
                ],
            }
        ]
    }
    h = compute_text_heuristics(page_dict)
    assert decide_route(h, mode="auto") == "unanchored"
    assert not is_text_only_candidate(h)


def test_printable_ratio_ignores_newline_separators():
    spans = [{"text": "Token"} for _ in range(220)]
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [{"spans": [span]} for span in spans],
            }
        ]
    }
    h = compute_text_heuristics(page_dict)
    assert h.printable_ratio > 0.99
    assert decide_route(h, mode="auto") == "anchored"
