# Private Telegram continuity (opt-in pilot)

The direct Browser Use provider accepts `browser.browser_use.profile_id` in the
owning Hermes profile's config. Provision one Browser Use profile per client;
provider project/key isolation must also be established before multi-client use.
Keep `browser.backend: "off"`, select `browser.cloud_provider: browser-use`, and
set `browser.inactivity_timeout: 600`. The existing native tools and vault remain
in use. Keep the API key in the profile's secret store, outside the image.

This opt-in path currently requires POSIX file locks and a private Telegram turn
whose authenticated chat and user IDs agree. Other surfaces must use the default
(nonpersistent) provider configuration. The local deployment launcher must support
the returned CDP endpoint.

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
