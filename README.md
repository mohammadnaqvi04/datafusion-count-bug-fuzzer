# subquery-fuzz-harness

A differential fuzzer for the DataFusion "count bug" in correlated
`EXISTS` / `NOT EXISTS` / `IN` / `NOT IN` subqueries over groupless
aggregates. It's built to test the CAST-collision fan-out fix.

Each round generates a random query and dataset, then runs it against three
binaries:

- **DuckDB**: the oracle (ground truth)
- **`main`**: a DataFusion CLI built from upstream `main`
- **branch**: a DataFusion CLI built from the branch under test

A case is reported when the branch's result differs from DuckDB's, or when
the branch errors on a query DuckDB runs fine. `main`'s result is printed
next to each finding for context only. `main` often gets these shapes wrong
in its own way, so it isn't used to filter findings.

## What it generates

Each case picks one option from each dimension:

| Dimension   | Options |
|-------------|---------|
| Correlation | `i.k = o.k`, or `CAST(i.k AS INT) = o.k` over TEXT keys where `'1'` and `'01'` collide |
| Aggregate   | `count(*)`, `sum(v)` |
| HAVING      | none, simple, NULL-producing, compound |
| Wrapper     | `EXISTS`, `NOT EXISTS`, `0 IN`, `0 NOT IN`, `EXISTS` inside an `OR` |
| CTE         | subquery used directly, or wrapped in a CTE (adds a `SubqueryAlias` boundary) |

## Usage

You need the `duckdb` CLI and two `datafusion-cli` builds, one from `main`
and one from your branch.

```sh
cargo run --release -- \
  --duckdb /path/to/duckdb \
  --main   /path/to/datafusion-cli-main \
  --branch /path/to/datafusion-cli-branch \
  [--seed 42] [--rounds 200]
```

Set `FUZZ_DEBUG=1` to log every case with all three results.

Run `cargo test` for the unit tests that cover result parsing.
