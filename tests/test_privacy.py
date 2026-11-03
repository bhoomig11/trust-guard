"""Privacy KPI: every mock PII value must be removed before inference."""

import pytest

from trustguard.privacy.shield import PrivacyShield, get_shield

# (text, raw values that must not survive, placeholders that must appear)
PII_CASES = [
    ("My SSN is 123-45-6789, can I raise my wire limit?", ["123-45-6789"], ["[SSN]"]),
    ("Ahmed Khan, SSN 987 65 4320, asked about fees", ["987 65 4320", "Ahmed", "Khan"], ["[SSN]"]),
    (
        "Customer John Smith wants to dispute a charge on account 4401928374.",
        ["John", "Smith", "4401928374"],
        ["[PERSON]", "[ACCOUNT_NUMBER]"],
    ),
    (
        "Maria Garcia's card 4111 1111 1111 1111 was charged a foreign fee",
        ["Maria", "Garcia", "4111 1111 1111 1111"],
        ["[CARD_NUMBER]"],
    ),
    ("Amex card 3782 822463 10005 was declined", ["3782 822463 10005"], ["[CARD_NUMBER]"]),
    ("Refund to acct #77-201934 please", ["77-201934"], ["[ACCOUNT_NUMBER]"]),
    ("Wire from account 000123456789 to the vendor", ["000123456789"], ["[ACCOUNT_NUMBER]"]),
    (
        "Email jane.doe@example.com or call 617-555-0143 about the overdraft",
        ["jane.doe@example.com", "617-555-0143"],
        ["[EMAIL]", "[PHONE_NUMBER]"],
    ),
    ("Wei Zhang asked what the daily ATM limit is", ["Wei Zhang"], ["[PERSON]"]),
    ("Can Olivia Brown get her late fee reversed?", ["Olivia", "Brown"], ["[PERSON]"]),
    ("Routing 021000021 account 99887766", ["021000021", "99887766"], ["[ACCOUNT_NUMBER]"]),
]

BENIGN_QUERIES = [
    "What is the late payment fee?",
    "How many days do we have to file a SAR?",
    "What is the CTR threshold for cash deposits over $10,000?",
    "If a customer wires $30,000 internationally, what fee applies?",
    "What is 2.7% of $1,250?",
    "How long are account records kept after an account is closed?",
    "Which account types have an ACH limit?",
]


@pytest.mark.parametrize(("raw", "secrets", "placeholders"), PII_CASES)
def test_pii_is_fully_masked(shield: PrivacyShield, raw, secrets, placeholders) -> None:
    result = shield.mask(raw)
    for secret in secrets:
        assert secret not in result.masked_text
    for placeholder in placeholders:
        assert placeholder in result.masked_text
    assert result.pii_detected


@pytest.mark.parametrize("query", BENIGN_QUERIES)
def test_benign_queries_pass_through(shield: PrivacyShield, query: str) -> None:
    result = shield.mask(query)
    assert result.masked_text == query
    assert not result.pii_detected


def test_interception_rate_is_100_percent(shield: PrivacyShield) -> None:
    leaked = [
        secret
        for raw, secrets, _ in PII_CASES
        for secret in secrets
        if secret in shield.mask(raw).masked_text
    ]
    assert leaked == []


def test_entity_metadata_never_contains_values(shield: PrivacyShield) -> None:
    result = shield.mask("SSN 123-45-6789")
    serialized = str([e.to_dict() for e in result.entities])
    assert "123-45-6789" not in serialized
    assert result.entities[0].entity_type == "US_SSN"


def test_empty_text(shield: PrivacyShield) -> None:
    assert shield.mask("   ").masked_text == "   "


def test_get_shield_is_a_singleton() -> None:
    assert get_shield() is get_shield()
