//! Standalone fuzzer for the DataFusion "count bug" fix in correlated
//! EXISTS/NOT EXISTS/IN/NOT IN subqueries over groupless aggregates.
//!
//! Generates random query shapes and data, runs each against three
//! binaries (DuckDB, a fresh `main` DataFusion build, and the branch
//! under test), and reports any case where the branch disagrees with
//! DuckDB. DuckDB serves as the ground truth. `main`'s own agreement or
//! disagreement with DuckDB is reported alongside each finding for
//! context, not used to decide whether to report it -- `main` often
//! has its own, different count-bug-related wrong answers for these
//! same shapes, so requiring `main` to be correct first would miss most
//! of the cases this fix actually needs to cover.
//!
//! Not wired into `cargo test` or the DataFusion workspace. Run directly:
//!   cargo run --release -- --duckdb <path> --main <path> --branch <path> [--seed N] [--rounds N]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone)]
struct GeneratedCase {
    setup_sql: String,
    query_sql: String,
    description: String,
}

struct Generator {
    rng: StdRng,
}

/// A correlation predicate, and the specific value(s) on the inner
/// table's side that are chosen so that two different rows collide
/// under it -- the actual trigger for the fan-out bug.
#[derive(Clone, Copy)]
enum CorrelationShape {
    /// `i.k = o.k`, inner column stored as the same type as the outer.
    Bare,
    /// `CAST(i.k AS INT) = o.k`, inner column stored as TEXT so two
    /// different raw values ('1', '01') can collide after the cast.
    CastCollision,
}

#[derive(Clone, Copy)]
enum Aggregate {
    CountStar,
    Sum,
}

#[derive(Clone, Copy)]
enum Having {
    None,
    Simple,
    NullProducing,
    Compound,
}

#[derive(Clone, Copy)]
enum Wrapper {
    Exists,
    NotExists,
    In,
    NotIn,
    EmbeddedInOr,
}

/// Whether the subquery's body sits directly under the wrapper, or one
/// `SubqueryAlias` boundary further out via a CTE. A correlated column
/// folded into a group-by alias (see the fix this harness targets) has
/// to carry that alias forward across a `SubqueryAlias` boundary too,
/// not just resolve directly under the `Aggregate` it was grouped in --
/// a case the plain (non-CTE) shapes below never exercise.
#[derive(Clone, Copy)]
enum CteWrapping {
    None,
    ViaCte,
}

impl Generator {
    fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn pick_correlation(&mut self) -> CorrelationShape {
        if self.rng.random_bool(0.5) {
            CorrelationShape::CastCollision
        } else {
            CorrelationShape::Bare
        }
    }

    fn pick_aggregate(&mut self) -> Aggregate {
        if self.rng.random_bool(0.5) {
            Aggregate::CountStar
        } else {
            Aggregate::Sum
        }
    }

    fn pick_having(&mut self) -> Having {
        match self.rng.random_range(0..4) {
            0 => Having::None,
            1 => Having::Simple,
            2 => Having::NullProducing,
            _ => Having::Compound,
        }
    }

    fn pick_wrapper(&mut self) -> Wrapper {
        match self.rng.random_range(0..5) {
            0 => Wrapper::Exists,
            1 => Wrapper::NotExists,
            2 => Wrapper::In,
            3 => Wrapper::NotIn,
            _ => Wrapper::EmbeddedInOr,
        }
    }

    fn pick_cte_wrapping(&mut self) -> CteWrapping {
        if self.rng.random_bool(0.5) {
            CteWrapping::ViaCte
        } else {
            CteWrapping::None
        }
    }

    /// Outer table has 2-4 rows; inner table's rows are shaped by the
    /// chosen correlation predicate so some outer rows have zero
    /// matches, some have one, and (for CastCollision) some outer rows
    /// have multiple colliding inner groups with different v values, so
    /// their per-group HAVING results can disagree with each other.
    fn generate_case(&mut self) -> GeneratedCase {
        let correlation = self.pick_correlation();
        let aggregate = self.pick_aggregate();
        let having = self.pick_having();
        let wrapper = self.pick_wrapper();
        let cte_wrapping = self.pick_cte_wrapping();

        let (inner_setup, outer_setup) = match correlation {
            CorrelationShape::Bare => {
                let v_third = self_or_null(&mut self.rng);
                let inner_values = [
                    "(1, 10)".to_string(),
                    "(1, 20)".to_string(),
                    match v_third {
                        Some(v) => format!("(2, {v})"),
                        None => "(2, NULL)".to_string(),
                    },
                ]
                .join(", ");
                let outer_values = [1, 2, 3]
                    .iter()
                    .map(|k| format!("({k})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                (
                    format!(
                        "CREATE TABLE fi(k INT, v INT);\nINSERT INTO fi VALUES {inner_values};"
                    ),
                    format!("CREATE TABLE fo(k INT);\nINSERT INTO fo VALUES {outer_values};"),
                )
            }
            CorrelationShape::CastCollision => {
                // '1' and '01' both cast to 1, with different v values
                // so their per-group HAVING outcomes can disagree; '2'
                // has a single, non-colliding match.
                let v1: i64 = self.rng.random_range(1..50);
                let v2: i64 = self.rng.random_range(1..50);
                let v3_null = self.rng.random_bool(0.5);
                let v3 = if v3_null {
                    "NULL".to_string()
                } else {
                    self.rng.random_range(1..50).to_string()
                };
                (
                    format!(
                        "CREATE TABLE fi(k TEXT, v INT);\nINSERT INTO fi VALUES ('1', {v1}), ('01', {v2}), ('2', {v3});"
                    ),
                    "CREATE TABLE fo(k INT);\nINSERT INTO fo VALUES (1), (2), (3);".to_string(),
                )
            }
        };

        let correlation_expr = match correlation {
            CorrelationShape::Bare => "i.k = o.k".to_string(),
            CorrelationShape::CastCollision => "CAST(i.k AS INT) = o.k".to_string(),
        };

        let agg_expr = match aggregate {
            Aggregate::CountStar => "count(*)".to_string(),
            Aggregate::Sum => "sum(v)".to_string(),
        };

        let having_clause = match (having, &aggregate) {
            (Having::None, _) => String::new(),
            (Having::Simple, Aggregate::CountStar) => " HAVING count(*) = 0".to_string(),
            (Having::Simple, Aggregate::Sum) => " HAVING sum(v) > 5".to_string(),
            (Having::NullProducing, Aggregate::CountStar) => {
                " HAVING count(*) = 0 OR sum(v) > 0".to_string()
            }
            (Having::NullProducing, Aggregate::Sum) => " HAVING sum(v) IS NULL".to_string(),
            (Having::Compound, Aggregate::CountStar) => {
                " HAVING count(*) = 0 AND count(*) < 100".to_string()
            }
            (Having::Compound, Aggregate::Sum) => {
                " HAVING sum(v) IS NULL OR sum(v) > 1000".to_string()
            }
        };

        let subquery = match cte_wrapping {
            CteWrapping::None => format!(
                "SELECT {agg_expr} FROM fi i WHERE {correlation_expr}{having_clause}"
            ),
            CteWrapping::ViaCte => format!(
                "WITH agg AS (SELECT {agg_expr} AS c FROM fi i WHERE {correlation_expr}{having_clause}) SELECT c FROM agg"
            ),
        };

        let query_sql = match wrapper {
            Wrapper::Exists => {
                format!("SELECT o.k FROM fo o WHERE EXISTS ({subquery}) ORDER BY o.k;")
            }
            Wrapper::NotExists => {
                format!("SELECT o.k FROM fo o WHERE NOT EXISTS ({subquery}) ORDER BY o.k;")
            }
            Wrapper::In => format!("SELECT o.k FROM fo o WHERE 0 IN ({subquery}) ORDER BY o.k;"),
            Wrapper::NotIn => {
                format!("SELECT o.k FROM fo o WHERE 0 NOT IN ({subquery}) ORDER BY o.k;")
            }
            Wrapper::EmbeddedInOr => {
                format!("SELECT o.k FROM fo o WHERE o.k > 100 OR EXISTS ({subquery}) ORDER BY o.k;")
            }
        };

        let description = format!(
            "correlation={}, aggregate={}, having={}, wrapper={}, cte={}",
            match correlation {
                CorrelationShape::Bare => "bare",
                CorrelationShape::CastCollision => "cast_collision",
            },
            match aggregate {
                Aggregate::CountStar => "count_star",
                Aggregate::Sum => "sum",
            },
            match having {
                Having::None => "none",
                Having::Simple => "simple",
                Having::NullProducing => "null_producing",
                Having::Compound => "compound",
            },
            match wrapper {
                Wrapper::Exists => "exists",
                Wrapper::NotExists => "not_exists",
                Wrapper::In => "in",
                Wrapper::NotIn => "not_in",
                Wrapper::EmbeddedInOr => "embedded_in_or",
            },
            match cte_wrapping {
                CteWrapping::None => "none",
                CteWrapping::ViaCte => "via_cte",
            },
        );

        GeneratedCase {
            setup_sql: format!("{inner_setup}\n{outer_setup}"),
            query_sql,
            description,
        }
    }
}

fn self_or_null(rng: &mut StdRng) -> Option<i64> {
    if rng.random_bool(0.5) {
        Some(rng.random_range(1..50))
    } else {
        None
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

fn run_datafusion_cli(binary: &PathBuf, sql_file: &PathBuf) -> Result<String, String> {
    let output = Command::new(binary)
        .arg("--format")
        .arg("csv")
        .arg("-q")
        .arg("-f")
        .arg(sql_file)
        .output()
        .map_err(|e| format!("failed to spawn datafusion-cli: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    // datafusion-cli in `-q -f` mode prints a failed statement's error to
    // STDERR and still exits 0 -- the exit code alone never catches this,
    // only a true process-level failure (spawn failure, signal). A
    // statement failing partway through the file is exactly the kind of
    // finding this harness exists to catch, so it has to be detected
    // from stderr's text, not just the exit code or stdout.
    if stderr.contains("Optimizer rule") && stderr.contains("failed") {
        return Err(stderr);
    }
    if stderr.contains("Schema error:")
        || stderr.contains("Internal error:")
        || stderr.contains("This feature is not implemented")
    {
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
/// line equal to `k` marks the start of the real result.
fn extract_query_result_rows(output: &str, header: &str) -> Vec<String> {
    let lines: Vec<&str> = output.lines().map(str::trim).collect();
    match lines.iter().rposition(|&l| l == header) {
        Some(idx) => lines[idx + 1..].iter().map(|s| s.to_string()).collect(),
        None => vec![],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut duckdb_path = None;
    let mut main_path = None;
    let mut branch_path = None;
    let mut seed: u64 = 42;
    let mut rounds: usize = 200;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--duckdb" => {
                duckdb_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--main" => {
                main_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--branch" => {
                branch_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("--seed must be a number");
                i += 2;
            }
            "--rounds" => {
                rounds = args[i + 1].parse().expect("--rounds must be a number");
                i += 2;
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    let duckdb_path = duckdb_path.expect("--duckdb <path> is required");
    let main_path = main_path.expect("--main <path> is required");
    let branch_path = branch_path.expect("--branch <path> is required");

    println!("Running {rounds} rounds with seed {seed}");
    println!("  duckdb: {}", duckdb_path.display());
    println!("  main:   {}", main_path.display());
    println!("  branch: {}", branch_path.display());
    println!();

    let mut gen = Generator::new(seed);
    let mut found_bugs = 0;
    let mut errors = 0;

    for round in 0..rounds {
        let case = gen.generate_case();
        let full_sql = format!("{}\n{}", case.setup_sql, case.query_sql);

        let scratch_file = std::env::temp_dir().join(format!("fuzz_case_{round}.sql"));
        std::fs::write(&scratch_file, &full_sql).expect("failed to write scratch sql file");

        // DuckDB is the oracle. If it can't even run the case, there's
        // nothing to compare against -- skip, this is a scratch-query
        // problem, not a finding.
        let duckdb_result = match run_duckdb(&duckdb_path, &full_sql) {
            Ok(out) => extract_query_result_rows(&out, "k"),
            Err(e) => {
                errors += 1;
                eprintln!("[round {round}] duckdb error ({}): {e}", case.description);
                continue;
            }
        };
        let main_result = match run_datafusion_cli(&main_path, &scratch_file) {
            Ok(out) => extract_query_result_rows(&out, "k"),
            Err(e) => {
                errors += 1;
                eprintln!("[round {round}] main error ({}): {e}", case.description);
                continue;
            }
        };
        // Unlike duckdb/main above, a branch error is not simply skipped:
        // DuckDB already proved the case is a valid query with a real
        // answer, so the branch crashing on it (rather than returning a
        // wrong result) is itself exactly the kind of finding this
        // harness exists to catch, not merely an execution error to
        // shrug off and move past.
        let branch_result = match run_datafusion_cli(&branch_path, &scratch_file) {
            Ok(out) => extract_query_result_rows(&out, "k"),
            Err(e) => {
                found_bugs += 1;
                println!("=== BRANCH ERRORED, DUCKDB SUCCEEDED [round {round}] ===");
                println!("  {}", case.description);
                println!("  setup: {}", case.setup_sql.replace('\n', " "));
                println!("  query: {}", case.query_sql);
                println!("  duckdb succeeded");
                println!("  branch error: {e}");
                println!();
                continue;
            }
        };

        let mut duckdb_sorted = duckdb_result.clone();
        duckdb_sorted.sort();
        let mut main_sorted = main_result.clone();
        main_sorted.sort();
        let mut branch_sorted = branch_result.clone();
        branch_sorted.sort();

        let duckdb_matches_main = duckdb_sorted == main_sorted;
        let branch_matches_duckdb = branch_sorted == duckdb_sorted;

        if std::env::var("FUZZ_DEBUG").is_ok() {
            eprintln!(
                "[round {round}] {} | duckdb={duckdb_sorted:?} main={main_sorted:?} branch={branch_sorted:?}\n  sql: {}",
                case.description,
                full_sql.replace('\n', " ")
            );
        }

        // DuckDB is the actual oracle here, not `main` -- `main` frequently
        // disagrees with DuckDB on its own for these shapes (that's the
        // count bug this PR exists to fix), so gating on `main` also
        // matching DuckDB silently skipped most of the shape space this
        // fix is supposed to cover. `main`'s agreement is reported
        // alongside purely as context, not as a precondition for flagging.
        if !branch_matches_duckdb {
            found_bugs += 1;
            println!("=== BRANCH DISAGREES WITH DUCKDB [round {round}] ===");
            println!("  {}", case.description);
            println!("  setup: {}", case.setup_sql.replace('\n', " "));
            println!("  query: {}", case.query_sql);
            println!("  duckdb: {duckdb_sorted:?}");
            println!(
                "  main:   {main_sorted:?}{}",
                if duckdb_matches_main {
                    " (matches duckdb)"
                } else {
                    " (ALSO disagrees with duckdb, for a possibly different reason)"
                }
            );
            println!("  branch: {branch_sorted:?}  <-- wrong");
            println!();
        }

        std::fs::remove_file(&scratch_file).ok();
    }

    println!("Done. {rounds} rounds, {found_bugs} regressions found, {errors} execution errors.");
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
}
