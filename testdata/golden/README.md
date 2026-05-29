# Golden Payload Fixtures

Provider payload golden tests compare canonical JSON under this directory. The
test helper normalizes JSON before comparison, so object key order and local Go
map iteration do not affect fixture diffs.

Reviewers should inspect golden changes as provider contract changes:

- Confirm added fields are supported by the provider API or compatibility
  profile covered by the test.
- Confirm removed fields are intentionally unsupported, deprecated, or covered
  by a compatibility flag.
- Check that fixture values are synthetic. Do not include real API keys,
  bearer tokens, account IDs, project IDs, customer data, or live provider
  responses.

To intentionally refresh fixtures after changing payload conversion:

```sh
UPDATE_GOLDEN=1 mise run go:test
mise run go:test
```

Do not use update mode to accept unexplained diffs. Golden tests supplement
behavior tests; they are not a replacement for provider behavior coverage.
