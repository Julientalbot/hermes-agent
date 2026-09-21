# Persistent browser continuity (opt-in)

The direct Browser Use provider accepts `browser.browser_use.profile_id` in the
owning Hermes profile's config. Provision one Browser Use profile per client;
provider project/key isolation must also be established before multi-client use.
Keep `browser.backend: "off"`, select `browser.cloud_provider: browser-use`, and
set `browser.inactivity_timeout: 600`. The existing native tools and vault remain
in use. Keep the API key in the profile's secret store, outside the image.

This opt-in path requires POSIX file locks. Telegram continuity follows the
authenticated conversation and topic, including authorized groups. Secret and
one-time-code collection remains private. Cron browsers are scoped to their
execution task and closed at turn completion; they may reuse profile cookies,
but cannot take over a different conversation’s live browser. Other native
surfaces use their runtime task identity. The launcher must support the CDP endpoint.

Hermes keeps its tracked browser between turns. Its existing inactivity reaper
closes it; the provider caps its life at thirty minutes. A private record under
`browser/cloud` stores the browser ID, owner fingerprint and timestamps, never the
CDP URL. A crash releases the OS lock. The same conversation can GET and reconnect
the active browser within ten minutes; a known stopped browser can be replaced.
Normal shutdown closes the browser; cookies can be reused through the profile,
but a login transaction interrupted by shutdown may need to be restarted.

A timed-out create leaves a `creating` record. Inspect the provider operation
metadata and resolve it before retrying; never delete that record simply to bypass
the exclusion. Configured persistent sessions never silently fall back to local
Chromium. CAPTCHA handling is a provider capability, not a success guarantee.
