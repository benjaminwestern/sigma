# Release notes: sigma v0.6.0

This is the maintainer-facing development note for the next `sigma` tag. Add
the v0.6.0 summary and scope as changes land. For the itemized change list see
[CHANGELOG.md](../CHANGELOG.md); for the validation commands and pre-tag
checklist see [RELEASING.md](../RELEASING.md).

## Release summary

`sigma` v0.6.0 opens with long prompt-cache usage accounting for providers
that report a separate long-lived cache-write split.

## Added

- Anthropic Messages usage now populates
  `sigma.Usage.LongCacheWriteInputTokens` from long prompt-cache write usage
  and `sigma.CostForUsage` prices those writes at the long-cache input
  multiplier while preserving total cache-write token accounting.

## Compatibility

- `sigma.Usage.LongCacheWriteInputTokens` is additive metadata for cost
  accounting. Existing `CacheWriteInputTokens` values remain the total cache
  write count, so callers that ignore the long-cache split keep the same token
  totals.

## Deferred work

- Deferred work continues to be tracked in [TODO.md](../TODO.md).

## Validation status

Validate this release with the process in [RELEASING.md](../RELEASING.md),
including the local CI-equivalent `mise run ci` gate before tagging.
