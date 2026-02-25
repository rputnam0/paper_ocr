from paper_ocr.inspect import (
    compute_text_heuristics,
    decide_route,
    is_reliable_text_layer_candidate,
    is_structured_page_candidate,
    is_text_only_candidate,
)


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


def test_structured_page_candidate_accepts_moderate_density_clean_text():
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"text": "123 456 789 table row"} for _ in range(25)]},
                ],
            }
        ]
    }
    h = compute_text_heuristics(page_dict)
    assert is_structured_page_candidate(h)
    assert not is_text_only_candidate(h)


def test_reliable_text_layer_candidate_accepts_clean_medium_density_page():
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"text": "Methods and results indicate stable convergence."}]},
                    {"spans": [{"text": " ".join(["token"] * 35)}]},
                ],
            }
        ]
    }
    h = compute_text_heuristics(page_dict)
    assert not is_text_only_candidate(h)
    assert is_reliable_text_layer_candidate(h)


def test_reliable_text_layer_candidate_rejects_cid_garbled_text():
    page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"text": "cid:101 cid:102 cid:103 " * 25}]},
                ],
            }
        ]
    }
    h = compute_text_heuristics(page_dict)
    assert not is_reliable_text_layer_candidate(h)
