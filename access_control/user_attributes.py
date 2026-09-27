"""The interface the retrieval/access-control layer expects the app's
own login/session system to supply. No users table, no auth is built
here - this is just the documented shape of the dict the app must hand
to vectorstore.retrieve.retrieve() and access_control.ask_and_answer()
once a real user is logged in.

    UserAttributes = {
        "country": str | None,          # e.g. "India" - None if unknown
        "employment_type": str | None,  # e.g. "full-time", "contractor" - None if unknown
    }

Keys correspond 1:1 to applicability_tags.tag_type values currently in
use (see access_control/extract_applicability.py). A key's value is
matched, case-sensitively, against applicability_tags.tag_value.

Passing None (or omitting this dict entirely) for the whole structure
means "unknown user" - no tag-based filtering is applied for any
attribute. Passing an explicit None for one key means "this user's
value for that attribute is unknown" - filtering for that specific
attribute is skipped, but other known attributes still filter normally,
and access_control.ask_and_answer() may ask a clarifying question if
that unknown attribute turns out to matter for the query at hand.
"""
from typing import Optional, TypedDict


class UserAttributes(TypedDict, total=False):
    country: Optional[str]
    employment_type: Optional[str]


KNOWN_TAG_TYPES = ("country", "employment_type")
