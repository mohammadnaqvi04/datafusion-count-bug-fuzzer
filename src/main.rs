//! Differential fuzzer for DataFusion's correlated-subquery "count bug"
//! fix, including the CAST-collision fan-out ('1' and '01' matching the
//! same outer row).
//!
//! Each round runs one generated case through DuckDB (ground truth), a
//! `main` build and the branch under test, and reports it if the branch
//! answers differently from DuckDB or errors. `main` is context only.
//!
//!   cargo run --release -- --duckdb <path> --main <path> --branch <path> [--seed N] [--rounds N]
//!   DUCKDB=<path> cargo test -- --ignored   # check cases against real DuckDB answers

use rand::rngs::StdRng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
struct GeneratedCase {
    shape: Shape,
    setup_sql: String,
    query_sql: String,
}

struct Generator {
    rng: StdRng,
    /// Shapes still to run in the current block of 160, in random order.
    pending: Vec<Shape>,
}

/// How the subquery is correlated with the outer row. Both shapes use
/// the same data, so for the same draw they have the same right answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CorrelationShape {
    /// `i.k = o.k`, inner column stored as the same type as the outer.
    /// The two inner rows for outer key 1 share the raw key `1`, so
    /// there is nothing to fan out: this is the count-bug control for
    /// the cast shape.
    Bare,
    /// `CAST(i.k AS INT) = o.k`, inner column stored as TEXT. The two
    /// inner rows for outer key 1 are '1' and '01': different raw values
    /// that collide after the cast. Grouping by the raw column instead
    /// of the cast expression splits them into two groups, each checked
    /// against HAVING on its own -- the fan-out bug.
    CastCollision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Aggregate {
    CountStar,
    Sum,
}

/// The subquery's HAVING clause. `n` and `t` come from the case's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Having {
    None,
    /// One comparison: `count(*) = n` / `sum(v) > t`.
    Simple,
    /// Mixes `count(*)`, which is never NULL, with `sum(v)`, which is
    /// NULL over no rows or all-NULL rows, so the predicate itself can
    /// evaluate to NULL: `count(*) = 0 OR sum(v) > t` /
    /// `sum(v) > t OR count(*) > 1`.
    NullProducing,
    /// Two disjuncts: `count(*) = n OR count(*) > 1` /
    /// `sum(v) IS NULL OR sum(v) < t`.
    Compound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Wrapper {
    Exists,
    NotExists,
    /// `x IN (subquery)`, with `x` drawn from values the subquery can
    /// return, so the comparison can actually match.
    In,
    NotIn,
    /// `o.k = m OR EXISTS (subquery)`: EXISTS inside a disjunction rather
    /// than as a top-level filter. `m` is 0 (no outer row), 2 or 3, so the
    /// OR lets a different row through depending on the draw; key 1, the
    /// colliding one, always depends on the subquery.
    EmbeddedInOr,
}

/// Whether the subquery's body sits directly under the wrapper, or one
/// `SubqueryAlias` boundary further out via a CTE. A correlated column
/// folded into a group-by alias (see the fix this harness targets) has
/// to carry that alias forward across a `SubqueryAlias` boundary too,
/// not just resolve directly under the `Aggregate` it was grouped in --
/// a case the plain (non-CTE) shapes below never exercise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CteWrapping {
    None,
    ViaCte,
}

/// The structural choices for one case; everything except the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Shape {
    correlation: CorrelationShape,
    aggregate: Aggregate,
    having: Having,
    wrapper: Wrapper,
    cte: CteWrapping,
}

impl Shape {
    /// Every combination of the five choices: 2 x 2 x 4 x 5 x 2 = 160.
    fn all() -> Vec<Shape> {
        let mut shapes = Vec::new();
        for correlation in [CorrelationShape::Bare, CorrelationShape::CastCollision] {
            for aggregate in [Aggregate::CountStar, Aggregate::Sum] {
                for having in [
                    Having::None,
                    Having::Simple,
                    Having::NullProducing,
                    Having::Compound,
                ] {
                    for wrapper in [
                        Wrapper::Exists,
                        Wrapper::NotExists,
                        Wrapper::In,
                        Wrapper::NotIn,
                        Wrapper::EmbeddedInOr,
                    ] {
                        for cte in [CteWrapping::None, CteWrapping::ViaCte] {
                            shapes.push(Shape {
                                correlation,
                                aggregate,
                                having,
                                wrapper,
                                cte,
                            });
                        }
                    }
                }
            }
        }
        shapes
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "correlation={}, aggregate={}, having={}, wrapper={}, cte={}",
            match self.correlation {
                CorrelationShape::Bare => "bare",
                CorrelationShape::CastCollision => "cast_collision",
            },
            match self.aggregate {
                Aggregate::CountStar => "count_star",
                Aggregate::Sum => "sum",
            },
            match self.having {
                Having::None => "none",
                Having::Simple => "simple",
                Having::NullProducing => "null_producing",
                Having::Compound => "compound",
            },
            match self.wrapper {
                Wrapper::Exists => "exists",
                Wrapper::NotExists => "not_exists",
                Wrapper::In => "in",
                Wrapper::NotIn => "not_in",
                Wrapper::EmbeddedInOr => "embedded_in_or",
            },
            match self.cte {
                CteWrapping::None => "none",
                CteWrapping::ViaCte => "via_cte",
            },
        )
    }
}

/// The random part of a case.
#[derive(Debug, Clone, Copy)]
struct CaseData {
    /// `v` for the two inner rows that both match outer key 1.
    v_a: Option<i64>,
    v_b: Option<i64>,
    /// `v` for the single inner row matching outer key 2.
    v_two: Option<i64>,
    /// `n` in `count(*) = n`.
    count_target: i64,
    /// `t` in the HAVING clauses' `sum(v)` comparisons.
    sum_threshold: i64,
    /// The left-hand side of IN / NOT IN.
    in_literal: i64,
    /// `m` in `o.k = m OR EXISTS (...)`.
    or_key: i64,
}

impl Generator {
    fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            pending: Vec::new(),
        }
    }

    fn pick_shape(&mut self) -> Shape {
        if self.pending.is_empty() {
            self.pending = Shape::all();
            self.pending.shuffle(&mut self.rng);
        }
        self.pending.pop().expect("Shape::all() is never empty")
    }

    /// Values are small, can be negative and are sometimes NULL. The sum
    /// threshold and the IN literal are each drawn from the sums the
    /// subquery can actually produce -- per row, combined for outer key 1,
    /// or 0 -- so the '1' row, the '01' row and their combined sum often
    /// land on different sides of the comparison. That's what makes a
    /// group split by fan-out produce a different answer.
    fn pick_data(&mut self, shape: &Shape) -> CaseData {
        let v_a = self.pick_v();
        let v_b = self.pick_v();
        let v_two = self.pick_v();
        let sums: Vec<i64> = [Some(0), v_a, v_b, v_two, sql_sum(v_a, v_b)]
            .into_iter()
            .flatten()
            .collect();
        let count_target = self.rng.random_range(0..=2);
        let sum_threshold = *sums.choose(&mut self.rng).expect("0 is always a candidate");
        let in_literal = match shape.aggregate {
            // Outer key 3 counts 0 rows, key 2 counts 1, key 1 counts 2
            // (or 1 per group once fan-out splits it).
            Aggregate::CountStar => self.rng.random_range(0..=2),
            Aggregate::Sum => *sums.choose(&mut self.rng).expect("0 is always a candidate"),
        };
        let or_key = *[0, 2, 3].choose(&mut self.rng).expect("not empty");
        CaseData {
            v_a,
            v_b,
            v_two,
            count_target,
            sum_threshold,
            in_literal,
            or_key,
        }
    }

    fn pick_v(&mut self) -> Option<i64> {
        if self.rng.random_bool(0.2) {
            None
        } else {
            Some(self.rng.random_range(-3..=6))
        }
    }

    fn generate_case(&mut self) -> GeneratedCase {
        let shape = self.pick_shape();
        let data = self.pick_data(&shape);
        render_case(&shape, &data)
    }
}

/// SQL's `sum` over two values: NULLs are skipped, and all-NULL is NULL.
fn sql_sum(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (None, None) => None,
        _ => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    }
}

fn sql_value(v: Option<i64>) -> String {
    match v {
        Some(v) => v.to_string(),
        None => "NULL".to_string(),
    }
}

/// The outer table is always (1), (2), (3): key 1 matches two inner rows
/// (the pair that collides under CastCollision), key 2 matches one, and
/// key 3 matches none, which is the empty group the count bug is about.
fn render_case(shape: &Shape, data: &CaseData) -> GeneratedCase {
    let (v_a, v_b, v_two) = (
        sql_value(data.v_a),
        sql_value(data.v_b),
        sql_value(data.v_two),
    );
    let inner_setup = match shape.correlation {
        CorrelationShape::Bare => format!(
            "CREATE TABLE fi(k INT, v INT);\nINSERT INTO fi VALUES (1, {v_a}), (1, {v_b}), (2, {v_two});"
        ),
        CorrelationShape::CastCollision => format!(
            "CREATE TABLE fi(k TEXT, v INT);\nINSERT INTO fi VALUES ('1', {v_a}), ('01', {v_b}), ('2', {v_two});"
        ),
    };
    let outer_setup = "CREATE TABLE fo(k INT);\nINSERT INTO fo VALUES (1), (2), (3);";

    let correlation_expr = match shape.correlation {
        CorrelationShape::Bare => "i.k = o.k".to_string(),
        CorrelationShape::CastCollision => "CAST(i.k AS INT) = o.k".to_string(),
    };

    let agg_expr = match shape.aggregate {
        Aggregate::CountStar => "count(*)".to_string(),
        Aggregate::Sum => "sum(v)".to_string(),
    };

    let (n, t) = (data.count_target, data.sum_threshold);
    let having_clause = match (shape.having, shape.aggregate) {
        (Having::None, _) => String::new(),
        (Having::Simple, Aggregate::CountStar) => format!(" HAVING count(*) = {n}"),
        (Having::Simple, Aggregate::Sum) => format!(" HAVING sum(v) > {t}"),
        (Having::NullProducing, Aggregate::CountStar) => {
            format!(" HAVING count(*) = 0 OR sum(v) > {t}")
        }
        (Having::NullProducing, Aggregate::Sum) => {
            format!(" HAVING sum(v) > {t} OR count(*) > 1")
        }
        (Having::Compound, Aggregate::CountStar) => {
            format!(" HAVING count(*) = {n} OR count(*) > 1")
        }
        (Having::Compound, Aggregate::Sum) => {
            format!(" HAVING sum(v) IS NULL OR sum(v) < {t}")
        }
    };

    let subquery = match shape.cte {
        CteWrapping::None => {
            format!("SELECT {agg_expr} FROM fi i WHERE {correlation_expr}{having_clause}")
        }
        CteWrapping::ViaCte => format!(
            "WITH agg AS (SELECT {agg_expr} AS c FROM fi i WHERE {correlation_expr}{having_clause}) SELECT c FROM agg"
        ),
    };

    let (x, m) = (data.in_literal, data.or_key);
    let query_sql = match shape.wrapper {
        Wrapper::Exists => {
            format!("SELECT o.k FROM fo o WHERE EXISTS ({subquery}) ORDER BY o.k;")
        }
        Wrapper::NotExists => {
            format!("SELECT o.k FROM fo o WHERE NOT EXISTS ({subquery}) ORDER BY o.k;")
        }
        Wrapper::In => format!("SELECT o.k FROM fo o WHERE {x} IN ({subquery}) ORDER BY o.k;"),
        Wrapper::NotIn => {
            format!("SELECT o.k FROM fo o WHERE {x} NOT IN ({subquery}) ORDER BY o.k;")
        }
        Wrapper::EmbeddedInOr => {
            format!("SELECT o.k FROM fo o WHERE o.k = {m} OR EXISTS ({subquery}) ORDER BY o.k;")
        }
    };

    GeneratedCase {
        shape: *shape,
        setup_sql: format!("{inner_setup}\n{outer_setup}"),
        query_sql,
    }
}

fn run_duckdb(binary: &PathBuf, sql: &str) -> Result<String, String> {
    let mut child = Command::new(binary)
        .arg(":memory:")
        .arg("-csv")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn duckdb: {e}"))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(sql.as_bytes())
        .map_err(|e| format!("failed to write to duckdb stdin: {e}"))?;
    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait on duckdb: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn run_datafusion_cli(binary: &PathBuf, sql_file: &Path) -> Result<String, String> {
    let output = Command::new(binary)
        .arg("--format")
        .arg("csv")
        .arg("-q")
        .arg("-f")
        .arg(sql_file)
        .output()
        .map_err(|e| format!("failed to spawn datafusion-cli: {e}"))?;
    check_datafusion_cli_output(
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// datafusion-cli in `-q -f` mode prints a failed statement's error to
/// STDERR, still exits 0, and carries on with the next statement. It
/// also prints nothing at all -- not even the header -- for a query with
/// no rows, so on stdout a failed query looks exactly like an empty
/// answer. stderr is the only signal, and a clean run leaves it empty,
/// so any stderr output counts as a failure: whatever the error's prefix
/// (`Arrow error:`, `Error during planning:`, `SQL error:`, ...) and
/// whichever statement, setup or query, it came from.
fn check_datafusion_cli_output(
    exited_ok: bool,
    stdout: String,
    stderr: String,
) -> Result<String, String> {
    if !exited_ok {
        return Err(format!("exited unsuccessfully: {stderr}"));
    }
    if !stderr.trim().is_empty() {
        return Err(stderr);
    }
    Ok(stdout)
}

/// Extract the final query's result rows from CSV output.
///
/// DuckDB in `-csv` mode is silent for `CREATE TABLE`/`INSERT` and only
/// prints the final `SELECT`'s header + rows, so its output is already
/// exactly what's wanted.
///
/// datafusion-cli in `--format csv` mode prints one CSV block per
/// statement, including a `count` block for each `INSERT`, so the
/// query's own block has to be picked out. Every query this harness
/// generates selects a single `k` column, and that header can't appear
/// as data (it's not a value any generated row produces), so the last
/// line equal to `k` marks the start of the real result. For a query
/// with no rows datafusion-cli prints no header at all, which the
/// no-header arm below turns into an empty result.
fn extract_query_result_rows(output: &str, header: &str) -> Vec<String> {
    let lines: Vec<&str> = output.lines().map(str::trim).collect();
    match lines.iter().rposition(|&l| l == header) {
        Some(idx) => lines[idx + 1..].iter().map(|s| s.to_string()).collect(),
        None => vec![],
    }
}

/// What one round amounts to once all three binaries have run.
#[derive(Debug, PartialEq)]
enum Verdict {
    /// DuckDB couldn't run the case, so there's no answer to check.
    Skipped,
    Agrees,
    /// The branch's answer differs from DuckDB's, and `main` doesn't get
    /// it right either (wrong answer or error): a bug the branch hasn't
    /// fixed, or has changed.
    BranchWrong,
    /// The branch's answer differs from DuckDB's while `main`'s matches:
    /// the branch broke something that worked.
    Regression,
    /// The branch errored on a case DuckDB answered.
    BranchErrored,
}

fn sorted(rows: &[String]) -> Vec<String> {
    let mut rows = rows.to_vec();
    rows.sort();
    rows
}

fn judge(
    duckdb: &Result<Vec<String>, String>,
    main: &Result<Vec<String>, String>,
    branch: &Result<Vec<String>, String>,
) -> Verdict {
    // DuckDB is the oracle. If it can't even run the case, there's
    // nothing to compare against -- skip, this is a scratch-query
    // problem, not a finding.
    let Ok(expected) = duckdb else {
        return Verdict::Skipped;
    };
    let expected = sorted(expected);
    // Unlike duckdb above, a branch error is not simply skipped: DuckDB
    // already proved the case is a valid query with a real answer, so
    // the branch crashing on it (rather than returning a wrong result)
    // is itself exactly the kind of finding this harness exists to
    // catch. `main` erroring changes nothing here; it only decides
    // whether a wrong branch answer counts as a regression.
    match branch {
        Err(_) => Verdict::BranchErrored,
        Ok(rows) if sorted(rows) == expected => Verdict::Agrees,
        Ok(_) => match main {
            Ok(rows) if sorted(rows) == expected => Verdict::Regression,
            _ => Verdict::BranchWrong,
        },
    }
}

/// One binary's result, as printed next to a finding.
fn describe(result: &Result<Vec<String>, String>, expected: &[String]) -> String {
    match result {
        Ok(rows) => {
            let rows = sorted(rows);
            let note = if rows == expected {
                "matches duckdb"
            } else {
                "differs from duckdb"
            };
            format!("{rows:?} ({note})")
        }
        Err(e) => format!("ERROR: {}", e.trim().replace('\n', " | ")),
    }
}

const USAGE: &str =
    "usage: --duckdb <path> --main <path> --branch <path> [--seed N (42)] [--rounds N (320)]";

#[derive(Debug)]
struct Args {
    duckdb: PathBuf,
    main: PathBuf,
    branch: PathBuf,
    seed: u64,
    rounds: usize,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let (mut duckdb, mut main, mut branch) = (None, None, None);
    let (mut seed, mut rounds) = (42, 320);
    let mut rest = args.iter().skip(1);
    while let Some(flag) = rest.next() {
        let value = rest.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--duckdb" => duckdb = Some(PathBuf::from(value)),
            "--main" => main = Some(PathBuf::from(value)),
            "--branch" => branch = Some(PathBuf::from(value)),
            "--seed" => seed = value.parse().map_err(|_| "--seed must be a number")?,
            "--rounds" => rounds = value.parse().map_err(|_| "--rounds must be a number")?,
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args {
        duckdb: duckdb.ok_or("--duckdb <path> is required")?,
        main: main.ok_or("--main <path> is required")?,
        branch: branch.ok_or("--branch <path> is required")?,
        seed,
        rounds,
    })
}

fn main() {
    let args = parse_args(&std::env::args().collect::<Vec<_>>()).unwrap_or_else(|e| {
        eprintln!("{e}\n{USAGE}");
        std::process::exit(2);
    });
    let Args {
        duckdb: duckdb_path,
        main: main_path,
        branch: branch_path,
        seed,
        rounds,
    } = args;

    println!("Running {rounds} rounds with seed {seed}");
    println!("  duckdb: {}", duckdb_path.display());
    println!("  main:   {}", main_path.display());
    println!("  branch: {}", branch_path.display());
    println!();

    let mut gen = Generator::new(seed);
    let (mut regressions, mut wrong, mut branch_errors) = (0, 0, 0);
    let (mut skipped, mut main_errors) = (0, 0);

    for round in 0..rounds {
        let case = gen.generate_case();
        let full_sql = format!("{}\n{}", case.setup_sql, case.query_sql);

        // The process id keeps two concurrent runs from overwriting each
        // other's files.
        let scratch_file =
            std::env::temp_dir().join(format!("fuzz_case_{}_{round}.sql", std::process::id()));
        std::fs::write(&scratch_file, &full_sql).expect("failed to write scratch sql file");
        let duckdb_result =
            run_duckdb(&duckdb_path, &full_sql).map(|out| extract_query_result_rows(&out, "k"));
        let main_result = run_datafusion_cli(&main_path, &scratch_file)
            .map(|out| extract_query_result_rows(&out, "k"));
        let branch_result = run_datafusion_cli(&branch_path, &scratch_file)
            .map(|out| extract_query_result_rows(&out, "k"));
        std::fs::remove_file(&scratch_file).ok();

        if std::env::var("FUZZ_DEBUG").is_ok() {
            eprintln!(
                "[round {round}] {} | duckdb={duckdb_result:?} main={main_result:?} branch={branch_result:?}\n  sql: {}",
                case.shape,
                full_sql.replace('\n', " ")
            );
        }
        if main_result.is_err() {
            main_errors += 1;
        }

        let verdict = judge(&duckdb_result, &main_result, &branch_result);
        let title = match verdict {
            Verdict::Agrees => continue,
            Verdict::Skipped => {
                skipped += 1;
                eprintln!(
                    "[round {round}] skipped, duckdb error ({}): {}",
                    case.shape,
                    duckdb_result.unwrap_err()
                );
                continue;
            }
            Verdict::Regression => {
                regressions += 1;
                "REGRESSION: BRANCH DISAGREES WITH DUCKDB, MAIN AGREES"
            }
            Verdict::BranchWrong => {
                wrong += 1;
                "BRANCH DISAGREES WITH DUCKDB"
            }
            Verdict::BranchErrored => {
                branch_errors += 1;
                "BRANCH ERRORED, DUCKDB SUCCEEDED"
            }
        };
        let expected = sorted(duckdb_result.as_ref().expect("judged, so duckdb succeeded"));
        println!("=== {title} [round {round}] ===");
        println!("  {}", case.shape);
        println!("  setup: {}", case.setup_sql.replace('\n', " "));
        println!("  query: {}", case.query_sql);
        println!("  duckdb: {expected:?}");
        println!("  main:   {}", describe(&main_result, &expected));
        println!(
            "  branch: {}  <-- wrong",
            describe(&branch_result, &expected)
        );
        println!();
    }

    let findings = regressions + wrong + branch_errors;
    println!(
        "Done. {rounds} rounds, {findings} findings: {regressions} regressions (main right, branch wrong), \
         {wrong} other wrong answers, {branch_errors} branch errors. \
         {skipped} rounds skipped (DuckDB error). main errored in {main_errors} rounds (context only)."
    );
    if findings > 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_datafusion_cli_csv_with_insert_count_noise() {
        // "count" blocks come from INSERT statements' own affected-row
        // output, and must not be mistaken for the query result.
        let output = "count\n2\ncount\n3\nk\n1\n2\n";
        assert_eq!(
            extract_query_result_rows(output, "k"),
            vec!["1".to_string(), "2".to_string()]
        );
    }

    #[test]
    fn test_extract_duckdb_csv_silent_setup() {
        let output = "k\n1\n2\n";
        assert_eq!(
            extract_query_result_rows(output, "k"),
            vec!["1".to_string(), "2".to_string()]
        );
    }

    #[test]
    fn test_extract_empty_result() {
        let output = "count\n2\nk\n";
        let result: Vec<String> = vec![];
        assert_eq!(extract_query_result_rows(output, "k"), result);
    }

    #[test]
    fn test_extract_no_header_found() {
        let output = "count\n2\n";
        let result: Vec<String> = vec![];
        assert_eq!(extract_query_result_rows(output, "k"), result);
    }

    #[test]
    fn test_datafusion_cli_statement_errors_are_detected_whatever_their_prefix() {
        // Real stderr from datafusion-cli 55.1 `-q -f`, which exited 0 for
        // every one of these.
        for stderr in [
            "Arrow error: Divide by zero error\n",
            "Arrow error: Cast error: Cannot cast string 'x' to value of Int32 type\n",
            "Error during planning: table 'datafusion.public.nope' not found\n",
            "SQL error: ParserError(\"Expected: an SQL statement, found: SELEC\")\n",
            "Execution error: something went wrong\n",
        ] {
            let result = check_datafusion_cli_output(true, "count\n1\n".into(), stderr.into());
            assert!(result.is_err(), "not detected as an error: {stderr}");
        }
    }

    #[test]
    fn test_datafusion_cli_clean_run_is_success() {
        assert_eq!(
            check_datafusion_cli_output(true, "count\n1\nk\n1\n".into(), String::new()),
            Ok("count\n1\nk\n1\n".to_string())
        );
    }

    fn rows(rows: &[&str]) -> Result<Vec<String>, String> {
        Ok(rows.iter().map(|r| r.to_string()).collect())
    }

    #[test]
    fn test_main_error_does_not_hide_wrong_branch_answer() {
        let main = Err("main fell over".to_string());
        assert_eq!(
            judge(&rows(&["1"]), &main, &rows(&[])),
            Verdict::BranchWrong
        );
    }

    #[test]
    fn test_main_error_does_not_hide_branch_error() {
        let main = Err("main fell over".to_string());
        let branch = Err("branch fell over".to_string());
        assert_eq!(judge(&rows(&["1"]), &main, &branch), Verdict::BranchErrored);
    }

    #[test]
    fn test_duckdb_error_skips_the_round() {
        let duckdb = Err("duckdb fell over".to_string());
        assert_eq!(judge(&duckdb, &rows(&[]), &rows(&[])), Verdict::Skipped);
    }

    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once("harness")
            .chain(args.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn test_flag_without_value_is_an_error_not_a_panic() {
        let args = argv(&["--duckdb", "d", "--main", "m", "--branch", "b", "--seed"]);
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_args_parse_with_defaults() {
        let args = parse_args(&argv(&["--duckdb", "d", "--main", "m", "--branch", "b"])).unwrap();
        assert_eq!(
            (args.duckdb, args.main, args.branch, args.seed, args.rounds),
            ("d".into(), "m".into(), "b".into(), 42, 320)
        );
    }

    #[test]
    fn test_every_shape_runs_once_per_160_rounds() {
        let mut gen = Generator::new(42);
        for block in 0..3 {
            let shapes: std::collections::HashSet<Shape> =
                (0..160).map(|_| gen.generate_case().shape).collect();
            assert_eq!(shapes.len(), 160, "block {block} missed some shapes");
        }
    }

    // ---- Checks against real DuckDB answers ------------------------------
    //
    // These run the generator's actual SQL through DuckDB to check that the
    // cases can tell right answers from wrong ones. Run them with
    //   DUCKDB=/path/to/duckdb cargo test -- --ignored

    /// Runs every (setup, query) pair in one DuckDB session and returns each
    /// query's result rows, sorted.
    fn duckdb_answers(cases: &[(String, String)]) -> Vec<Vec<String>> {
        let mut script = String::new();
        for (i, (setup, query)) in cases.iter().enumerate() {
            script.push_str(&format!(
                "DROP TABLE IF EXISTS fi;\nDROP TABLE IF EXISTS fo;\n{setup}\n.print ###{i}\n{query}\n"
            ));
        }
        let duckdb = std::env::var("DUCKDB").unwrap_or_else(|_| "duckdb".to_string());
        let out = run_duckdb(&PathBuf::from(&duckdb), &script)
            .unwrap_or_else(|e| panic!("duckdb failed (set DUCKDB=/path/to/duckdb): {e}"));
        let mut answers: Vec<Vec<String>> = Vec::new();
        for line in out.lines() {
            if line.starts_with("###") {
                answers.push(Vec::new());
            } else if line != "k" {
                answers
                    .last_mut()
                    .expect("output before marker")
                    .push(line.to_string());
            }
        }
        assert_eq!(answers.len(), cases.len(), "duckdb skipped some queries");
        answers.iter().map(|a| sorted(a)).collect()
    }

    fn setup_and_query(case: &GeneratedCase) -> (String, String) {
        (case.setup_sql.clone(), case.query_sql.clone())
    }

    /// The fan-out this harness targets: the HAVING check runs once per raw
    /// inner key ('1' and '01' separately) instead of once per outer row.
    /// Modelled by grouping the subquery by the raw key.
    fn with_fan_out(query: &str) -> String {
        let correlation = "CAST(i.k AS INT) = o.k";
        assert!(
            query.contains(correlation),
            "not a cast-collision query: {query}"
        );
        query.replacen(correlation, &format!("{correlation} GROUP BY i.k"), 1)
    }

    #[test]
    #[ignore = "needs DuckDB: DUCKDB=/path/to/duckdb cargo test -- --ignored"]
    fn test_cast_collision_cases_can_expose_fan_out_under_every_wrapper() {
        let mut gen = Generator::new(42);
        let cases: Vec<GeneratedCase> = (0..1600)
            .map(|_| gen.generate_case())
            .filter(|c| c.shape.correlation == CorrelationShape::CastCollision)
            .collect();
        let correct = duckdb_answers(&cases.iter().map(setup_and_query).collect::<Vec<_>>());
        let fanned = duckdb_answers(
            &cases
                .iter()
                .map(|c| (c.setup_sql.clone(), with_fan_out(&c.query_sql)))
                .collect::<Vec<_>>(),
        );
        for wrapper in [
            Wrapper::Exists,
            Wrapper::NotExists,
            Wrapper::In,
            Wrapper::NotIn,
            Wrapper::EmbeddedInOr,
        ] {
            let idx: Vec<usize> = (0..cases.len())
                .filter(|&i| cases[i].shape.wrapper == wrapper)
                .collect();
            // Only outer key 1 has colliding inner rows; keys 2 and 3 differ
            // under the model for count-bug reasons, not fan-out.
            let exposed = idx
                .iter()
                .filter(|&&i| correct[i].contains(&"1".into()) != fanned[i].contains(&"1".into()))
                .count();
            let share = exposed as f64 / idx.len() as f64;
            assert!(
                share >= 0.25,
                "{wrapper:?}: only {exposed}/{} cast-collision cases can expose fan-out",
                idx.len()
            );
        }
    }

    #[test]
    #[ignore = "needs DuckDB: DUCKDB=/path/to/duckdb cargo test -- --ignored"]
    fn test_answers_depend_on_the_random_data() {
        let mut gen = Generator::new(42);
        // 30 draws per shape.
        let cases: Vec<GeneratedCase> = (0..4800).map(|_| gen.generate_case()).collect();
        let answers = duckdb_answers(&cases.iter().map(setup_and_query).collect::<Vec<_>>());
        let mut seen: std::collections::HashMap<Shape, std::collections::HashSet<Vec<String>>> =
            Default::default();
        for (case, answer) in cases.iter().zip(answers) {
            seen.entry(case.shape).or_default().insert(answer);
        }
        let invariant: Vec<&Shape> = seen
            .iter()
            .filter(|(shape, answers)| {
                // A groupless aggregate with no HAVING returns exactly one
                // row whatever the data, so EXISTS / NOT EXISTS over it are
                // constant by definition.
                let constant_by_definition = shape.having == Having::None
                    && matches!(
                        shape.wrapper,
                        Wrapper::Exists | Wrapper::NotExists | Wrapper::EmbeddedInOr
                    );
                !constant_by_definition && answers.len() < 2
            })
            .map(|(shape, _)| shape)
            .collect();
        assert!(
            invariant.is_empty(),
            "{} shapes give the same answer for every data draw, e.g. {:?}",
            invariant.len(),
            invariant.first()
        );
    }

    #[test]
    #[ignore = "needs DuckDB: DUCKDB=/path/to/duckdb cargo test -- --ignored"]
    fn test_having_variants_are_never_equivalent() {
        let havings = [
            Having::None,
            Having::Simple,
            Having::NullProducing,
            Having::Compound,
        ];
        let mut gen = Generator::new(42);
        let mut cases = Vec::new();
        let mut groups = Vec::new();
        for correlation in [CorrelationShape::Bare, CorrelationShape::CastCollision] {
            for aggregate in [Aggregate::CountStar, Aggregate::Sum] {
                for wrapper in [
                    Wrapper::Exists,
                    Wrapper::NotExists,
                    Wrapper::In,
                    Wrapper::NotIn,
                    Wrapper::EmbeddedInOr,
                ] {
                    let base = Shape {
                        correlation,
                        aggregate,
                        having: Having::None,
                        wrapper,
                        cte: CteWrapping::None,
                    };
                    let start = cases.len();
                    for _ in 0..40 {
                        let data = gen.pick_data(&base);
                        for having in havings {
                            let shape = Shape { having, ..base };
                            cases.push(setup_and_query(&render_case(&shape, &data)));
                        }
                    }
                    groups.push((base, start));
                }
            }
        }
        let answers = duckdb_answers(&cases);
        let mut equivalent = Vec::new();
        for (base, start) in groups {
            for a in 0..havings.len() {
                for b in a + 1..havings.len() {
                    let differs = (0..40).any(|draw| {
                        let row = start + draw * havings.len();
                        answers[row + a] != answers[row + b]
                    });
                    if !differs {
                        equivalent.push((
                            base.correlation,
                            base.aggregate,
                            base.wrapper,
                            havings[a],
                            havings[b],
                        ));
                    }
                }
            }
        }
        assert!(
            equivalent.is_empty(),
            "{} HAVING pairs never differ, e.g. {:?}",
            equivalent.len(),
            equivalent.first()
        );
    }
}
