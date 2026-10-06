# HR chatbot API

```bash
python3 -m vectorstore.setup_db   # idempotent; creates query_logs
./run_api.sh                      # http://localhost:8000/docs
python3 -m eval.api_smoke_test    # end-to-end check (server running)
```

```bash
B=http://localhost:8000
# fresh query
curl -s $B/api/chat -H 'content-type: application/json' \
  -d '{"query":"How many days of leave do I get?"}'
# -> {"type":"needs_clarification","blocking_tag_type":"country","original_query":...}
# resolve it (send back original_query + blocking_tag_type, and any user_attributes returned)
curl -s $B/api/chat -H 'content-type: application/json' \
  -d '{"query":"I work in India","pending_clarification":{"original_query":"How many days of leave do I get?","blocking_tag_type":"country"}}'
# upload / list / restrict
curl -s -F file=@"hr docs/Travel Policy_Ver1.0.pdf" $B/api/admin/documents/upload
curl -s $B/api/admin/documents
curl -s -X PATCH $B/api/admin/documents/<id>/restrict -H 'content-type: application/json' -d '{"is_restricted":true}'
# usage (days=0 means all time)
curl -s "$B/api/admin/usage/summary?days=7"
curl -s "$B/api/admin/usage/recent?limit=50"
```

Response `type` is `answer`, `refusal_shortcut` (low-confidence canned reply), or
`needs_clarification`. All include `user_attributes` - send them back on the next turn.
There is no auth (see the security note in `api/main.py`).
