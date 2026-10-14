"""Shared by the eval suites that count generation calls. stream_completion calls client.with_options(...) per attempt,
so the scoped client it gets back must be counted too. A retry of the SAME generation (after a real first-token timeout)
reuses the same `messages` object and is not counted again, so counts stay deterministic when the network is slow."""


def wrap_with_options(client, counted):
    """counted(create) -> a create() replacement that counts. Applied to every client.with_options(...) result."""
    original_with_options = client.with_options
    last_messages = [None]

    def with_options(*args, **kwargs):
        scoped = original_with_options(*args, **kwargs)
        raw = scoped.chat.completions.create
        counting = counted(raw)

        def create(*a, **k):
            if k.get("messages") is last_messages[0]:
                return raw(*a, **k)
            last_messages[0] = k.get("messages")
            return counting(*a, **k)

        scoped.chat.completions.create = create
        return scoped

    client.with_options = with_options
