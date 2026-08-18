# run.R
# Known-variance DS&S ESS-Gibbs sampler for manifest-based MCMC runs.
# Output is restricted to two paper-ready files:
#   mcmc_summary.csv
#   mcmc_pip.csv

rm(list = ls())

# ============================================================
# 1. Original elliptical slice sampling within Gibbs
#    Do not modify this block.
# ============================================================

ESS.Gibbs <- function(
  b.c = rep(0, length(sd.0)),
  LL,
  id,
  sd.0,
  N = 5000,
  S.max = 500
) {
  d    <- length(b.c)
  K    <- length(id)
  v    <- sapply(id, length)
  N.s  <- matrix(NA_real_, N, K)
  mc.b <- matrix(NA_real_, N, d)

  for (i in seq_len(N)) {
    for (k in seq_len(K)) {
      n.s <- 0
      l   <- LL(b.c) + log(runif(1))
      th  <- runif(1, 0, 2 * pi)
      a   <- th - 2 * pi
      A   <- th
      nu  <- rnorm(v[k], 0, sd.0[id[[k]]])
      b.p <- b.c

      while (n.s < S.max) {
        b.p[id[[k]]] <- b.c[id[[k]]] * cos(th) + nu * sin(th)
        n.s <- n.s + 1

        if (LL(b.p) > l) {
          b.c <- b.p
          break
        }

        if (th < 0) {
          a <- th
        } else {
          A <- th
        }

        th <- runif(1, a, A)
      }

      N.s[i, k] <- n.s
    }

    mc.b[i, ] <- b.c
  }

  list(mc.b = mc.b, n.s = N.s)
}


# ============================================================
# 2. Data, split, and metrics
# ============================================================

make_split <- function(n, split_seed, train_frac = 0.6, val_frac = 0.2) {
  set.seed(as.integer(split_seed))

  idx <- sample.int(n)
  n_train <- floor(train_frac * n)
  n_val <- floor(val_frac * n)
  n_test <- n - n_train - n_val

  train_idx <- idx[seq_len(n_train)]
  val_idx <- idx[(n_train + 1L):(n_train + n_val)]
  test_idx <- idx[(n_train + n_val + 1L):n]

  list(
    train = train_idx,
    val = val_idx,
    test = test_idx,
    n_train = n_train,
    n_val = n_val,
    n_test = n_test
  )
}

read_data_csv <- function(job, split_seed) {
  dat <- read.csv(job$data_path, check.names = FALSE)
  beta_tbl <- read.csv(job$beta_path, check.names = FALSE)

  y_full <- dat$y
  X_full <- as.matrix(dat[, setdiff(names(dat), "y"), drop = FALSE])
  beta_true <- as.numeric(beta_tbl$beta_true)

  n <- nrow(X_full)
  p <- ncol(X_full)

  if (length(beta_true) != p) {
    stop("Length of beta_true does not match ncol(X).")
  }

  split <- make_split(n, split_seed = split_seed)

  list(
    X = X_full[split$train, , drop = FALSE],
    y = y_full[split$train],
    beta_true = beta_true,
    b0 = beta_true,
    n = n,
    p = p,
    n_train = split$n_train,
    n_val = split$n_val,
    n_test = split$n_test
  )
}

safe_auroc <- function(score, truth) {
  truth <- as.integer(truth)
  n_pos <- sum(truth == 1L)
  n_neg <- sum(truth == 0L)

  if (n_pos == 0L || n_neg == 0L) {
    return(NA_real_)
  }

  r <- rank(score, ties.method = "average")
  (sum(r[truth == 1L]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

safe_auprc <- function(score, truth) {
  truth <- as.integer(truth)
  n_pos <- sum(truth == 1L)

  if (n_pos == 0L) {
    return(NA_real_)
  }

  ord <- order(score, decreasing = TRUE)
  y <- truth[ord]
  tp <- cumsum(y == 1L)
  fp <- cumsum(y == 0L)
  precision <- tp / pmax(tp + fp, 1L)

  sum(precision[y == 1L]) / n_pos
}

selection_metrics <- function(selected, truth, score) {
  selected <- as.integer(selected)
  truth <- as.integer(truth)

  tp <- sum(selected == 1L & truth == 1L)
  fp <- sum(selected == 1L & truth == 0L)
  fn <- sum(selected == 0L & truth == 1L)
  tn <- sum(selected == 0L & truth == 0L)

  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
  fdr <- ifelse(tp + fp > 0, fp / (tp + fp), 0)

  data.frame(
    precision = precision,
    recall = recall,
    f1 = f1,
    fdr = fdr,
    tp = tp,
    fp = fp,
    fn = fn,
    tn = tn,
    support_size = sum(selected),
    auroc = safe_auroc(score, truth),
    auprc = safe_auprc(score, truth)
  )
}


# ============================================================
# 3. Extract practical PIP and write the two tables
# ============================================================

write_mcmc_tables <- function(
  MC,
  job,
  dat,
  burnin,
  thin,
  beta_eps = 0.5,
  pip_threshold = 0.5,
  split_seed = 12345,
  out_dir = "."
) {
  B <- as.matrix(MC$mc.b)
  B <- B[complete.cases(B), , drop = FALSE]

  if (nrow(B) <= burnin) {
    stop("Not enough valid MCMC draws after burnin.")
  }

  p <- dat$p
  keep <- seq.int(burnin + 1L, nrow(B), by = thin)
  draws <- B[keep, , drop = FALSE]

  theta <- t(apply(draws, 1, function(b) {
    V  <- b[seq_len(p)]
    W  <- b[p + seq_len(p)]
    W0 <- b[2L * p + 1L]
    V * pmax(W - W0, 0)
  }))

  theta_tbl <- as.data.frame(theta)
  names(theta_tbl) <- paste0("b", seq_len(p) - 1L)
  theta_tbl <- cbind(draw_id = seq_len(nrow(theta_tbl)) - 1L, theta_tbl)

  active_draw <- abs(theta) > beta_eps
  pip <- colMeans(active_draw)
  selected <- as.integer(pip > pip_threshold)
  effect_mean <- colMeans(theta)

  beta_true <- dat$beta_true
  truth <- as.integer(abs(beta_true) > 1e-12)

  pip_tbl <- data.frame(
    method = "mcmc",
    setting = job$setting,
    seed = as.integer(job$seed),
    replicate_id = as.integer(job$seed),
    j0 = seq_len(p) - 1L,
    j1 = seq_len(p),
    beta_true = beta_true,
    truth = truth,
    pip = pip,
    selected = selected,
    effect_mean = effect_mean
  )

  metr <- selection_metrics(selected, truth, pip)

  summary_tbl <- data.frame(
    method = "mcmc",
    setting = job$setting,
    seed = as.integer(job$seed),
    replicate_id = as.integer(job$seed),
    n = dat$n,
    p = dat$p,
    n_train = dat$n_train,
    n_val = dat$n_val,
    n_test = dat$n_test,
    n_active = sum(truth),
    sigma2 = ifelse("sigma2" %in% names(job) && !is.na(job$sigma2), as.numeric(job$sigma2), 1.0),
    split_seed = as.integer(split_seed),
    burnin = burnin,
    thin = thin,
    n_kept = nrow(draws),
    beta_eps = beta_eps,
    pip_threshold = pip_threshold,
    mean_support_size = mean(rowSums(active_draw)),
    mean_pip = mean(pip),
    selected_size = sum(selected),
    active_idx0 = paste(which(truth == 1L) - 1L, collapse = ";"),
    selected_idx0 = paste(which(selected == 1L) - 1L, collapse = ";")
  )

  summary_tbl <- cbind(summary_tbl, metr)

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  write.csv(
    pip_tbl,
    file = file.path(out_dir, "mcmc_pip.csv"),
    row.names = FALSE
  )

  write.csv(
    summary_tbl,
    file = file.path(out_dir, "mcmc_summary.csv"),
    row.names = FALSE
  )

  write.csv(
    theta_tbl,
    file = gzfile(file.path(out_dir, "mcmc_beta_draws.csv.gz")),
    row.names = FALSE
  )

  list(pip = pip_tbl, summary = summary_tbl)
}


# ============================================================
# 4. Run one manifest row
# ============================================================

run_mcmc_one <- function(
  job,
  N,
  S.max,
  burnin,
  thin,
  beta_eps = 0.5,
  split_seed = 12345,
  pip_threshold = 0.5
) {
  seed <- as.integer(job$seed)
  set.seed(seed)

  dat <- read_data_csv(job, split_seed = split_seed)

  X <- dat$X
  Y <- dat$y
  beta_true <- dat$beta_true
  b0 <- dat$b0
  p <- dat$p

  sigma2 <- ifelse("sigma2" %in% names(job) && !is.na(job$sigma2), as.numeric(job$sigma2), 1.0)
  sigma <- sqrt(sigma2)

  T.n <- function(b) pmax(b, 0)

  b.c <- c(b0, rep(1, p), 0)
  sd.0 <- sqrt(c(rep(1, p), rep(1, p), 1))

  L.exa <- function(b) {
    V  <- b[seq_len(p)]
    W  <- b[p + seq_len(p)]
    W0 <- b[2L * p + 1L]
    theta <- V * T.n(W - W0)

    sum(dnorm(
      x = Y,
      mean = as.numeric(X %*% theta),
      sd = sigma,
      log = TRUE
    ))
  }

  id <- lapply(seq_along(b.c), function(i) i)
  dir.create(job$out_dir, recursive = TRUE, showWarnings = FALSE)

  MC <- ESS.Gibbs(
    b.c = b.c,
    LL = L.exa,
    id = id,
    sd.0 = sd.0,
    N = N,
    S.max = S.max
  )

  write_mcmc_tables(
    MC = MC,
    job = job,
    dat = dat,
    burnin = burnin,
    thin = thin,
    beta_eps = beta_eps,
    pip_threshold = pip_threshold,
    split_seed = split_seed,
    out_dir = job$out_dir
  )
}


# ============================================================
# 5. Command-line interface
# ============================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2L) {
  stop("Usage: Rscript run.R manifest_mcmc.csv row_id [N] [Smax] [burnin] [thin] [beta_eps] [split_seed]")
}

manifest_path <- args[[1]]
row_id <- as.integer(args[[2]])

N_mcmc <- ifelse(length(args) >= 3L, as.integer(args[[3]]), 10000L)
S_max  <- ifelse(length(args) >= 4L, as.integer(args[[4]]), 100L)
burnin <- ifelse(length(args) >= 5L, as.integer(args[[5]]), 2000L)
thin   <- ifelse(length(args) >= 6L, as.integer(args[[6]]), 1L)
beta_eps <- ifelse(length(args) >= 7L, as.numeric(args[[7]]), 0.05)
split_seed <- ifelse(length(args) >= 8L, as.integer(args[[8]]), 12345L)
pip_threshold <- 0.5

manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
job <- manifest[row_id, ]

cat("[info] method: MCMC\n")
cat("[info] row_id:", row_id, "\n")
cat("[info] setting:", job$setting, "\n")
cat("[info] seed:", job$seed, "\n")
cat("[info] split_seed:", split_seed, "\n")
cat("[info] data:", job$data_path, "\n")
cat("[info] beta:", job$beta_path, "\n")
cat("[info] out_dir:", job$out_dir, "\n")
cat("[info] N:", N_mcmc, "S.max:", S_max, "burnin:", burnin, "thin:", thin, "\n")
cat("[info] beta_eps:", beta_eps, "\n")
cat("[info] pip_threshold:", pip_threshold, "\n")
cat("[info] variance: known\n")

out <- invisible(run_mcmc_one(
  job = job,
  N = N_mcmc,
  S.max = S_max,
  burnin = burnin,
  thin = thin,
  beta_eps = beta_eps,
  split_seed = split_seed,
  pip_threshold = pip_threshold
))

cat("[done] wrote mcmc_summary.csv and mcmc_pip.csv\n")
