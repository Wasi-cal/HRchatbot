"""ONE place that drops every cache derived from the document set. Call it on every upload, status change, title
edit, restrict toggle and tag change (never invalidate one of these by hand):

    topic list            (small-talk "what can you do", no-match reply)      config.topics
    catalog title list    ("what documents do you have")                      config.topics
    label norms           (which tag sets are the norm, so which chunks get a label)   config.labels
    duplicate scan cache  (admin documents list badges)                       vectorstore.duplicates
    in-process copies of the above in this process (other workers' copies expire within 30s)

Tag caches: tags are read with each chunk's details in the retrieval join, so there is no separate per-document
tag cache; the label norms are the only tag-derived cache.
"""
from config.labels import invalidate_label_norms
from config.topics import invalidate_topics
from vectorstore.duplicates import invalidate_scan_cache


def invalidate_document_caches() -> None:
    invalidate_topics()
    invalidate_label_norms()
    invalidate_scan_cache()
