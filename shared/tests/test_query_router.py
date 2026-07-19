from shared.query_router import classify, normalize


def test_normalize_strips_case_punctuation_and_whitespace():
    assert normalize("  Hi!! ") == "hi"
    assert normalize("Good   Morning.") == "good morning"


def test_exact_match_greeting_no_prior_context():
    result = classify("hi", has_prior_context=False)
    assert result.tier == "exact"
    assert result.intent == "greeting"
    assert result.score == 1.0
    assert result.would_shortcircuit is True
    assert result.context_gated is False


def test_exact_match_is_case_and_punctuation_insensitive():
    result = classify("Hello!", has_prior_context=False)
    assert result.tier == "exact"
    assert result.intent == "greeting"


def test_fuzzy_match_catches_typo():
    result = classify("helo", has_prior_context=False)
    assert result.tier == "fuzzy"
    assert result.intent == "greeting"
    assert result.would_shortcircuit is True


def test_thanks_and_farewell_intents():
    assert classify("thanks", has_prior_context=False).intent == "thanks"
    assert classify("bye", has_prior_context=False).intent == "farewell"


def test_no_match_for_real_query():
    result = classify("what were our top 5 products last quarter", has_prior_context=False)
    assert result.tier == "none"
    assert result.intent is None
    assert result.would_shortcircuit is False


def test_bare_acknowledgment_shortcircuits_only_as_first_message():
    # "no" is deliberately in the canned list (see CANNED_INTENTS comment)
    # - it's only safe to fast-path when there's nothing for it to be a
    # follow-up to, i.e. no prior context.
    first_turn = classify("no", has_prior_context=False)
    assert first_turn.intent == "acknowledgment"
    assert first_turn.would_shortcircuit is True

    follow_up = classify("no", has_prior_context=True)
    assert follow_up.intent == "acknowledgment"
    assert follow_up.context_gated is True
    assert follow_up.would_shortcircuit is False


def test_bot_identity_and_status_smalltalk_intents():
    assert classify("what's your name", has_prior_context=False).intent == "bot_identity"
    assert classify("how are you", has_prior_context=False).intent == "status_smalltalk"


def test_context_gating_blocks_shortcircuit_even_on_match():
    # Same query as the exact-match greeting test, but now there's a prior
    # turn in the chat - a short reply here is more likely a follow-up than
    # small talk, so it should still classify (for logging) but never be
    # marked safe to short-circuit.
    result = classify("hi", has_prior_context=True)
    assert result.intent == "greeting"
    assert result.context_gated is True
    assert result.would_shortcircuit is False


def test_long_message_skips_matching_tiers_entirely():
    long_query = "hi there, can you please help me clean up this messy spreadsheet"
    result = classify(long_query, has_prior_context=False)
    assert result.tier == "none"
    assert result.intent is None


def test_classify_never_raises_on_bad_input():
    # None is not a valid query, but classify() must degrade gracefully
    # rather than blow up the caller (see chats.py - shadow logging must
    # never affect the real request path).
    result = classify(None, has_prior_context=False)  # type: ignore[arg-type]
    assert result.tier == "none"
    assert result.error is not None
