# subquery-fuzz-harness

A differential fuzzer for the DataFusion "count bug" in correlated
`EXISTS` / `NOT EXISTS` / `IN` / `NOT IN` subqueries over groupless
aggregates. It includes the CAST-collision fan-out case, where
`CAST(i.k AS INT) = o.k` makes two different inner keys (`'1'` and `'01'`)
match the same outer row.

Each round generates a query and its data, then runs it against three
binaries:

- **DuckDB**: the oracle (ground truth)
- **`main`**: a DataFusion CLI built from upstream `main`
- **branch**: a DataFusion CLI built from the branch under test

A round is reported when the branch returns a different answer from
DuckDB's, or errors on a query DuckDB answers. `main` never decides whether
a round is reported. Its result, or its error, is printed next to each
finding as context, and it marks a wrong answer as a **regression** when
`main` got it right. A round is skipped only if DuckDB itself fails.

Any output on `datafusion-cli`'s stderr counts as an error. `datafusion-cli
-q -f` exits 0 when a statement fails, and it prints nothing at all for an
empty result, so stderr is the only way to tell a failed query from an
empty answer.

## What it generates

There are 160 query shapes, one for each combination below. Every block of
160 rounds runs each shape once, in random order.

| Dimension   | Options |
|-------------|---------|
| Correlation | `i.k = o.k` (control), or `CAST(i.k AS INT) = o.k` over TEXT keys where `'1'` and `'01'` collide |
| Aggregate   | `count(*)`, `sum(v)` |
| HAVING      | none; simple (`count(*) = n`, `sum(v) > t`); NULL-producing (`count(*) = 0 OR sum(v) > t`, `sum(v) > t OR count(*) > 1`); compound (`count(*) = n OR count(*) > 1`, `sum(v) IS NULL OR sum(v) < t`) |
| Wrapper     | `EXISTS`, `NOT EXISTS`, `x IN`, `x NOT IN`, `o.k = m OR EXISTS` |
| CTE         | subquery used directly, or wrapped in a CTE (adds a `SubqueryAlias` boundary) |

The outer table is always keys 1, 2 and 3. Key 1 has two inner rows (the
colliding pair), key 2 has one, and key 3 has none (the empty group). The
inner `v` values are random: small, sometimes negative, sometimes NULL. The
thresholds `t` and the IN literal `x` are drawn from the sums the subquery
can actually produce. As a result, the colliding rows often land on
different sides of the comparison, which is what exposes a group split by
fan-out. With the default seed, 57 of the 320 rounds are cases where that
fan-out changes the answer. Both correlations use the same data, so the
`i.k = o.k` shape is an exact control for the CAST one.

## Usage

You need the `duckdb` CLI and two `datafusion-cli` builds, one from `main`
and one from your branch.

```sh
cargo run --release -- \
  --duckdb /path/to/duckdb \
  --main   /path/to/datafusion-cli-main \
  --branch /path/to/datafusion-cli-branch \
  [--seed 42] [--rounds 320]
```

The harness exits 1 if it reports any finding, 2 for bad arguments, and 0
otherwise. Set `FUZZ_DEBUG=1` to log every round's SQL and all three
results.

## Tests

- `cargo test` checks the harness's own logic: result parsing, error
  detection, how each round is judged, argument parsing, and shape
  coverage.
- `DUCKDB=/path/to/duckdb cargo test -- --ignored` also runs the generated
  cases through DuckDB. It checks three things:
  - Under every wrapper, at least a quarter of the CAST cases can expose
    the fan-out.
  - Each shape's answer depends on the random data, except where SQL makes
    it constant.
  - No two HAVING options are equivalent.
