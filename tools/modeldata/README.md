# Model metadata generation

Sigma stores built-in model metadata in `internal/modeldata/catalog.json`.
The file is a curated snapshot, not a live provider query. Pricing values are
only as accurate as the snapshot date and source URLs recorded in the catalog.

Refresh flow:

1. Update `internal/modeldata/catalog.json`.
2. Run `mise run go:generate`.
3. Review `models_generated.go`, `image_models_generated.go`, and
   `embedding_models_generated.go`.
4. Run `mise run go:test`.

The generator validates required fields and emits text, image, and embedding
models ordered by provider, API, and model ID. Generated files are deterministic
and should not be edited by hand.

For a local catalog summary while regenerating, run:

```bash
mise run go:run -- ./cmd/sigma-generate-models -report
```

The summary is reporting-only. It does not fetch provider catalogs or change the
checked-in catalog.
