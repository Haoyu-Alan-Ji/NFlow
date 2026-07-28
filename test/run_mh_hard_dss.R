# Eight-chain known-variance hard-DSS component-wise random-walk MH.
# Chains run sequentially. Each scalar proposal recomputes the full posterior.

rm(list = ls())

if (!requireNamespace("posterior", quietly = TRUE)) {
  stop("Install the CRAN package 'posterior' before running this script.")
}

make_split <- function(n, split_seed, train_frac = 0.6, val_frac = 0.2) {
  set.seed(as.integer(split_seed))
  idx <- sample.int(n)
  n_train <- floor(train_frac * n)
  n_val <- floor(val_frac * n)

  list(
    train = idx[seq_len(n_train)],
    val = idx[(n_train + 1L):(n_train + n_val)],
    test = idx[(n_train + n_val + 1L):n],
    n_train = n_train,
    n_val = n_val,
    n_test = n - n_train - n_val
  )
}

read_data_csv <- function(job, split_seed) {
  dat <- read.csv(job$data_path, check.names = FALSE)
  beta_tbl <- read.csv(job$beta_path, check.names = FALSE)
  X_full <- as.matrix(dat[, setdiff(names(dat), "y"), drop = FALSE])
  split <- make_split(nrow(X_full), split_seed)

  list(
    X = X_full[split$train, , drop = FALSE],
    y = dat$y[split$train],
    beta_true = as.numeric(beta_tbl$beta_true),
    n = nrow(X_full),
    p = ncol(X_full),
    n_train = split$n_train,
    n_val = split$n_val,
    n_test = split$n_test
  )
}

new_chain <- function(b.c, log_post, proposal_scale) {
  list(
    b = b.c,
    lp = log_post(b.c),
    proposal_sd = rep(proposal_scale, length(b.c)),
    iter = 0L,
    accept_total = integer(length(b.c)),
    adapt_accept = integer(length(b.c)),
    adapt_count = 0L,
    runtime_s = 0
  )
}

run_chunk <- function(
  chain,
  log_post,
  n_iter,
  warmup,
  adapt_every = 100L,
  target_accept = 0.44
) {
  d <- length(chain$b)
  draws <- matrix(NA_real_, n_iter, d)
  t0 <- proc.time()[["elapsed"]]

  for (i in seq_len(n_iter)) {
    accepted <- integer(d)

    for (k in seq_len(d)) {
      b.p <- chain$b
      b.p[k] <- rnorm(1L, chain$b[k], chain$proposal_sd[k])
      lp.p <- log_post(b.p)

      if (is.finite(lp.p) && log(runif(1L)) < lp.p - chain$lp) {
        chain$b <- b.p
        chain$lp <- lp.p
        accepted[k] <- 1L
      }
    }

    chain$iter <- chain$iter + 1L
    chain$accept_total <- chain$accept_total + accepted
    chain$adapt_accept <- chain$adapt_accept + accepted
    chain$adapt_count <- chain$adapt_count + 1L
    draws[i, ] <- chain$b

    if (
      chain$iter <= warmup &&
      chain$iter %% adapt_every == 0L
    ) {
      rate <- chain$adapt_accept / chain$adapt_count
      step <- 0.5 / sqrt(chain$iter / adapt_every)
      chain$proposal_sd <- chain$proposal_sd * exp(step * (rate - target_accept))
      chain$proposal_sd <- pmin(pmax(chain$proposal_sd, 1e-4), 2)
      chain$adapt_accept[] <- 0L
      chain$adapt_count <- 0L
    }
  }

  chain$runtime_s <- chain$runtime_s + proc.time()[["elapsed"]] - t0
  list(chain = chain, draws = draws)
}

convergence_diagnostics <- function(samples, p) {
  n_chain <- length(samples)
  n_draw <- min(vapply(samples, nrow, integer(1)))
  variables <- c(
    paste0("V[", seq_len(p), "]"),
    paste0("W[", seq_len(p), "]"),
    "W0",
    "model_size"
  )

  if (n_draw < 4L) {
    return(data.frame(
      variable = variables,
      rhat = Inf,
      ess_bulk = 0,
      ess_tail = 0
    ))
  }

  x <- array(
    NA_real_,
    dim = c(n_draw, n_chain, length(variables)),
    dimnames = list(NULL, paste0("chain", seq_len(n_chain)), variables)
  )

  for (chain_id in seq_len(n_chain)) {
    z <- tail(samples[[chain_id]], n_draw)
    W <- z[, p + seq_len(p), drop = FALSE]
    W0 <- z[, 2L * p + 1L]
    model_size <- rowSums(sweep(W, 1L, W0, FUN = ">"))
    x[, chain_id, ] <- cbind(z, model_size)
  }

  out <- posterior::summarise_draws(
    posterior::as_draws_array(x),
    "rhat",
    "ess_bulk",
    "ess_tail"
  )
  out <- as.data.frame(out)

  required <- c(
    "variable",
    "rhat",
    "ess_bulk",
    "ess_tail"
  )

  if (!all(required %in% names(out))) {
    stop(
      "Unexpected posterior::summarise_draws columns: ",
      paste(names(out), collapse = ", ")
    )
  }

  for (j in seq_along(variables)) {
    row <- match(variables[j], out$variable)

    if (is.na(row)) {
      stop(
        "Diagnostic variable not returned: ",
        variables[j]
      )
    }

    if (
      variables[j] == "model_size" &&
      all(x[, , j] == x[1L, 1L, j])
    ) {
      out$rhat[row] <- 1
      out$ess_bulk[row] <- n_draw * n_chain
      out$ess_tail[row] <- n_draw * n_chain
    } else {
      if (!is.finite(out$rhat[row])) out$rhat[row] <- Inf
      if (!is.finite(out$ess_bulk[row])) out$ess_bulk[row] <- 0
      if (!is.finite(out$ess_tail[row])) out$ess_tail[row] <- 0
    }
  }

  out
}

run_mh_one <- function(
  job,
  n_chains = 8L,
  warmup = 10000L,
  check_every = 5000L,
  max_iter = 200000L,
  proposal_scale = 0.10,
  rhat_limit = 1.01,
  ess_limit = 400,
  diagnostic_window = 10000L,
  split_seed = 12345L
) {
  dat <- read_data_csv(job, split_seed)
  p <- dat$p
  d <- 2L * p + 1L
  sigma2 <- if (
    "sigma2" %in% names(job) && !is.na(job$sigma2)
  ) as.numeric(job$sigma2) else 1

  log_post <- function(b) {
    V <- b[seq_len(p)]
    W <- b[p + seq_len(p)]
    W0 <- b[2L * p + 1L]
    theta <- V * as.numeric(W > W0)
    mu <- as.numeric(dat$X %*% theta)

    -0.5 * (
      sum((dat$y - mu)^2) / sigma2 +
      dat$n_train * log(2 * pi * sigma2) +
      sum(b^2) +
      d * log(2 * pi)
    )
  }

  chains <- vector("list", n_chains)
  samples <- vector("list", n_chains)

  for (chain_id in seq_len(n_chains)) {
    set.seed(as.integer(job$seed) + 100000L * chain_id)
    b.c <- rnorm(2L * p + 1L)
    chains[[chain_id]] <- new_chain(b.c, log_post, proposal_scale)
    samples[[chain_id]] <- matrix(numeric(0), nrow = 0L, ncol = d)
  }

  history <- list()
  diagnostics <- NULL
  converged <- FALSE
  total_t0 <- proc.time()[["elapsed"]]

  while (chains[[1L]]$iter < max_iter && !converged) {
    n_now <- min(check_every, max_iter - chains[[1L]]$iter)

    for (chain_id in seq_len(n_chains)) {
      before <- chains[[chain_id]]$iter
      chunk <- run_chunk(
        chain = chains[[chain_id]],
        log_post = log_post,
        n_iter = n_now,
        warmup = warmup
      )
      chains[[chain_id]] <- chunk$chain

      keep <- which(before + seq_len(n_now) > warmup)
      if (length(keep)) {
        samples[[chain_id]] <- rbind(
          samples[[chain_id]],
          chunk$draws[keep, , drop = FALSE]
        )
        samples[[chain_id]] <- tail(samples[[chain_id]], diagnostic_window)
      }

      cat(
        "[mh] chain", chain_id,
        "iteration", chains[[chain_id]]$iter,
        "acceptance", round(mean(chains[[chain_id]]$accept_total / chains[[chain_id]]$iter), 3),
        "\n"
      )
    }

    n_post <- min(vapply(samples, nrow, integer(1)))
    if (n_post >= check_every) {
      diagnostics <- convergence_diagnostics(samples, p)
      max_rhat <- max(diagnostics$rhat)
      min_bulk <- min(diagnostics$ess_bulk)
      min_tail <- min(diagnostics$ess_tail)
      converged <- (
        max_rhat <= rhat_limit &&
        min_bulk >= ess_limit &&
        min_tail >= ess_limit
      )

      history[[length(history) + 1L]] <- data.frame(
        iter_per_chain = chains[[1L]]$iter,
        postwarmup_per_chain = n_post,
        max_rhat = max_rhat,
        min_ess_bulk = min_bulk,
        min_ess_tail = min_tail,
        converged = converged
      )

      cat(
        "[diagnostics] iter", chains[[1L]]$iter,
        "Rhat_max", round(max_rhat, 4),
        "bulk_ESS_min", round(min_bulk, 1),
        "tail_ESS_min", round(min_tail, 1),
        "converged", converged,
        "\n"
      )
    }
  }

  total_runtime_s <- proc.time()[["elapsed"]] - total_t0
  mh_runtime_s <- sum(vapply(chains, function(x) x$runtime_s, numeric(1)))
  n_iter <- chains[[1L]]$iter
  n_post <- min(vapply(samples, nrow, integer(1)))

  if (is.null(diagnostics)) {
    diagnostics <- convergence_diagnostics(samples, p)
  }

  summary <- data.frame(
    method = "naive_componentwise_rwm",
    setting = job$setting,
    seed = as.integer(job$seed),
    n = dat$n,
    p = p,
    n_train = dat$n_train,
    n_chains = n_chains,
    initialization = "independent_prior_draws",
    execution = "sequential_chunks",
    n_iter_per_chain = n_iter,
    warmup_per_chain = warmup,
    postwarmup_total_per_chain = max(n_iter - warmup, 0L),
    diagnostic_draws_per_chain = n_post,
    diagnostic_window_per_chain = diagnostic_window,
    thinning = 1L,
    proposal_scale_initial = proposal_scale,
    mean_acceptance = mean(vapply(
      chains,
      function(x) mean(x$accept_total / x$iter),
      numeric(1)
    )),
    rhat_limit = rhat_limit,
    ess_bulk_limit = ess_limit,
    ess_tail_limit = ess_limit,
    max_rhat = max(diagnostics$rhat),
    min_ess_bulk = min(diagnostics$ess_bulk),
    min_ess_tail = min(diagnostics$ess_tail),
    converged = converged,
    stopping_reason = if (converged) "diagnostic_targets_met" else "max_iter_reached",
    full_log_posterior_evaluations = as.numeric(n_chains) * (
      1 + as.numeric(n_iter) * d
    ),
    mh_runtime_s = mh_runtime_s,
    diagnostics_runtime_s = total_runtime_s - mh_runtime_s,
    total_runtime_s = total_runtime_s
  )

  chain_summary <- do.call(rbind, lapply(seq_len(n_chains), function(chain_id) {
    x <- chains[[chain_id]]
    data.frame(
      chain = chain_id,
      seed = as.integer(job$seed) + 100000L * chain_id,
      n_iter = x$iter,
      mean_acceptance = mean(x$accept_total / x$iter),
      mean_proposal_sd = mean(x$proposal_sd),
      runtime_s = x$runtime_s
    )
  }))

  dir.create(job$out_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(summary, file.path(job$out_dir, "mh_runtime_summary.csv"), row.names = FALSE)
  write.csv(chain_summary, file.path(job$out_dir, "mh_chain_summary.csv"), row.names = FALSE)
  write.csv(diagnostics, file.path(job$out_dir, "mh_final_diagnostics.csv"), row.names = FALSE)
  write.csv(
    if (length(history)) do.call(rbind, history) else data.frame(),
    file.path(job$out_dir, "mh_diagnostic_history.csv"),
    row.names = FALSE
  )

  summary
}

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3L) {
  stop(
    paste(
      "Usage: Rscript run.mh.traditional.R manifest row_id output_root",
      "[n_chains] [warmup] [check_every] [max_iter]",
      "[proposal_scale] [rhat_limit] [ess_limit] [diagnostic_window] [split_seed]"
    )
  )
}

manifest <- read.csv(args[[1L]], stringsAsFactors = FALSE)
job <- manifest[as.integer(args[[2L]]), ]
output_root <- args[[3L]]
job$out_dir <- file.path(
  output_root,
  as.character(job$setting),
  paste0("seed_", as.integer(job$seed))
)

n_chains <- if (length(args) >= 4L) as.integer(args[[4L]]) else 8L
warmup <- if (length(args) >= 5L) as.integer(args[[5L]]) else 10000L
check_every <- if (length(args) >= 6L) as.integer(args[[6L]]) else 5000L
max_iter <- if (length(args) >= 7L) as.integer(args[[7L]]) else 200000L
proposal_scale <- if (length(args) >= 8L) as.numeric(args[[8L]]) else 0.10
rhat_limit <- if (length(args) >= 9L) as.numeric(args[[9L]]) else 1.01
ess_limit <- if (length(args) >= 10L) as.numeric(args[[10L]]) else 400
diagnostic_window <- if (length(args) >= 11L) as.integer(args[[11L]]) else 10000L
split_seed <- if (length(args) >= 12L) as.integer(args[[12L]]) else 12345L

if (n_chains < 2L) stop("At least two chains are required for R-hat.")
if (warmup < 0L || check_every < 1L || max_iter <= warmup) stop("Invalid iteration settings.")
if (diagnostic_window < check_every) stop("diagnostic_window must be at least check_every.")

cat("[info] method: eight-chain naive component-wise RWM\n")
cat("[info] chains run sequentially; thinning: 1\n")
cat("[info] setting:", job$setting, "seed:", job$seed, "\n")
cat("[info] n_chains:", n_chains, "warmup:", warmup, "check_every:", check_every, "\n")
cat("[info] max_iter:", max_iter, "Rhat <=", rhat_limit, "ESS >=", ess_limit, "\n")
cat("[info] diagnostic_window:", diagnostic_window, "post-warmup draws per chain\n")
cat("[info] out_dir:", job$out_dir, "\n")

result <- run_mh_one(
  job = job,
  n_chains = n_chains,
  warmup = warmup,
  check_every = check_every,
  max_iter = max_iter,
  proposal_scale = proposal_scale,
  rhat_limit = rhat_limit,
  ess_limit = ess_limit,
  diagnostic_window = diagnostic_window,
  split_seed = split_seed
)

print(result)