#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(torch)
  library(LBBNN)
  library(jsonlite)
})

parse_args <- function() {
  a <- commandArgs(trailingOnly = TRUE)
  out <- list(
    data_dir = NULL,
    out = "saha_r_results.csv",
    methods = "lbbnn_lrt,lbbnn_flow,islab_flow,is_ann_l1",
    seed = 100400L,
    h1 = 20L,
    h2 = 10L,
    batch = 128L,
    device = "auto",
    epochs = 2000L,
    lr = 0.01,
    draws = 500L,
    prior = 0.5,
    prior_sd = 1.0,
    inclusion_inits = "polarized",
    num_transforms = 2L,
    flow_dims = "50,50",
    ann_epochs = 2000L,
    ann_lr = 0.01,
    ann_lambda = 0.01,
    ann_threshold = 0.005,
    verbose = FALSE
  )
  if (length(a) == 0) return(out)
  i <- 1L
  while (i <= length(a)) {
    key <- sub("^--", "", a[[i]])
    key <- gsub("-", "_", key)
    if (key == "verbose") {
      out[[key]] <- TRUE
      i <- i + 1L
    } else {
      if (i == length(a)) stop("Missing value for --", key)
      out[[key]] <- a[[i + 1L]]
      i <- i + 2L
    }
  }
  ints <- c("seed", "h1", "h2", "batch", "epochs", "draws", "num_transforms", "ann_epochs")
  nums <- c("lr", "prior", "prior_sd", "ann_lr", "ann_lambda", "ann_threshold")
  for (k in ints) out[[k]] <- as.integer(out[[k]])
  for (k in nums) out[[k]] <- as.numeric(out[[k]])
  out
}

resolve_bench_device <- function(device) {
  d <- tolower(device)
  if (d == "auto") return(if (torch::cuda_is_available()) "cuda" else "cpu")
  if (d == "gpu") return("cuda")
  d
}

read_vec <- function(path) as.numeric(read.csv(path, header = FALSE, check.names = FALSE)[[1]])
read_mat <- function(path) as.matrix(read.csv(path, header = FALSE, check.names = FALSE))

load_data <- function(data_dir) {
  d <- normalizePath(data_dir, mustWork = TRUE)
  req <- c("X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv", "signal_test.csv", "feature_true.csv")
  missing <- req[!file.exists(file.path(d, req))]
  if (length(missing)) stop("Missing files in data_dir: ", paste(missing, collapse = ", "))
  meta <- list(seed = NA_integer_)
  if (file.exists(file.path(d, "meta.json"))) meta <- jsonlite::read_json(file.path(d, "meta.json"), simplifyVector = TRUE)
  list(
    Xtr = read_mat(file.path(d, "X_train.csv")),
    ytr = read_vec(file.path(d, "y_train.csv")),
    Xte = read_mat(file.path(d, "X_test.csv")),
    yte = read_vec(file.path(d, "y_test.csv")),
    signal = read_vec(file.path(d, "signal_test.csv")),
    truth = read_vec(file.path(d, "feature_true.csv")) > 0.5,
    meta = meta
  )
}

pred_metrics <- function(pred, signal, y) {
  pred <- as.numeric(pred); signal <- as.numeric(signal); y <- as.numeric(y)
  mse_signal <- mean((pred - signal)^2)
  sst <- sum((signal - mean(signal))^2)
  r2 <- 1 - sum((pred - signal)^2) / max(sst, 1e-12)
  mse_y <- mean((pred - y)^2)
  c(mse_signal = mse_signal, r2_signal = r2, mse_y = mse_y, rmse_y = sqrt(mse_y))
}

selection_metrics <- function(selected, truth) {
  selected <- as.logical(selected); truth <- as.logical(truth)
  if (length(selected) != length(truth)) stop("selected feature mask has wrong length")
  tp <- sum(selected & truth); fp <- sum(selected & !truth)
  fn <- sum(!selected & truth); tn <- sum(!selected & !truth)
  c(
    tpr = tp / max(tp + fn, 1),
    fpr = fp / max(fp + tn, 1),
    accuracy = (tp + tn) / max(length(truth), 1),
    selected_support = sum(selected)
  )
}

make_loaders <- function(dat, batch, device) {
  xt <- torch_tensor(dat$Xtr, dtype = torch_float())
  yt <- torch_tensor(dat$ytr, dtype = torch_float())
  xv <- torch_tensor(dat$Xte, dtype = torch_float())
  yv <- torch_tensor(dat$yte, dtype = torch_float())
  list(
    train = dataloader(tensor_dataset(xt, yt), batch_size = min(batch, nrow(dat$Xtr)), shuffle = TRUE),
    test = dataloader(tensor_dataset(xv, yv), batch_size = nrow(dat$Xte), shuffle = FALSE)
  )
}

mean_prediction <- function(draw_tensor) {
  a <- as.array(draw_tensor$detach()$cpu())
  if (length(dim(a)) == 3L) return(as.numeric(apply(a, c(2, 3), mean)[, 1]))
  if (length(dim(a)) == 2L) return(as.numeric(colMeans(a)))
  as.numeric(a)
}

count_lbbnn_active_weights <- function(model) {
  retained <- 0
  candidate <- 0
  for (l in model$layers$children) {
    a <- l$alpha_active_path$detach()$cpu()
    retained <- retained + a$sum()$item()
    candidate <- candidate + a$numel()
  }
  a <- model$out_layer$alpha_active_path$detach()$cpu()
  retained <- retained + a$sum()$item()
  candidate <- candidate + a$numel()
  c(retained = retained, candidate = candidate)
}

lbbnn_selected_features <- function(model, p, input_skip) {
  if (input_skip) {
    inc <- LBBNN:::get_input_inclusions(model)
    return(as.logical(rowSums(inc) > 0))
  }
  first <- model$layers$children[[1]]$alpha_active_path$detach()$cpu()
  ap <- as.array(first)
  sel <- colSums(ap > 0) > 0
  if (length(sel) != p) stop("Could not recover p input selections from first LBBNN layer")
  as.logical(sel)
}

run_lbbnn <- function(method, dat, opt, device) {
  input_skip <- identical(method, "islab_flow")
  use_flow <- method %in% c("lbbnn_flow", "islab_flow")
  p <- ncol(dat$Xtr)
  sizes <- c(p, opt$h1, opt$h2, 1L)
  prior <- rep(opt$prior, length(sizes) - 1L)
  stds <- rep(opt$prior_sd, length(sizes) - 1L)
  flow_dims <- as.integer(strsplit(opt$flow_dims, ",", fixed = TRUE)[[1]])
  dl <- make_loaders(dat, opt$batch, device)
  torch_manual_seed(opt$seed)
  model <- lbbnn_net(
    problem_type = "regression", sizes = sizes, prior = prior, std = stds,
    inclusion_inits = opt$inclusion_inits, input_skip = input_skip, flow = use_flow,
    num_transforms = opt$num_transforms, dims = flow_dims, device = device,
    bias_inclusion_prob = FALSE, weight_init = "he"
  )
  t0 <- proc.time()[[3]]
  suppressMessages(train_lbbnn(
    epochs = opt$epochs, LBBNN = model, lr = opt$lr,
    train_dl = dl$train, device = device, verbose = opt$verbose
  ))
  runtime <- proc.time()[[3]] - t0
  # train_lbbnn computes active paths at the end; recompute to make this explicit.
  if (input_skip) model$compute_paths_input_skip() else model$compute_paths()
  draws <- predict(model, newdata = dl$test, mpm = TRUE, draws = opt$draws, device = device)
  pred <- mean_prediction(draws)
  selected <- lbbnn_selected_features(model, p, input_skip)
  counts <- count_lbbnn_active_weights(model)
  dparam <- counts[["retained"]] / counts[["candidate"]]
  native_density <- as.numeric(model$density())
  list(
    pred = pred, selected = selected,
    retained = counts[["retained"]], candidate = counts[["candidate"]],
    dparam = dparam, native_density = native_density, runtime = runtime,
    rule = "MPM PIP > 0.5; Dparam counts only weights lying on complete active paths"
  )
}

ISANNL1 <- nn_module(
  "ISANNL1",
  initialize = function(p, h1, h2) {
    self$p <- p; self$h1 <- h1; self$h2 <- h2
    self$fc1 <- nn_linear(p, h1)
    self$fc2 <- nn_linear(h1 + p, h2)
    self$out <- nn_linear(h2 + p, 1)
  },
  forward = function(x) {
    h1 <- nnf_relu(self$fc1(x))
    h2 <- nnf_relu(self$fc2(torch_cat(list(h1, x), dim = 2)))
    self$out(torch_cat(list(h2, x), dim = 2))$squeeze(2)
  }
)

ann_l1_penalty <- function(model) {
  model$fc1$weight$abs()$sum() + model$fc2$weight$abs()$sum() + model$out$weight$abs()$sum()
}

ann_masks_active_paths <- function(model, threshold) {
  w1 <- abs(as.array(model$fc1$weight$detach()$cpu())) >= threshold
  w2 <- abs(as.array(model$fc2$weight$detach()$cpu())) >= threshold
  w3 <- abs(as.array(model$out$weight$detach()$cpu())) >= threshold
  p <- model$p; h1 <- model$h1; h2 <- model$h2
  w2_h <- w2[, seq_len(h1), drop = FALSE]
  w2_x <- w2[, h1 + seq_len(p), drop = FALSE]
  w3_h <- w3[, seq_len(h2), drop = FALSE]
  w3_x <- w3[, h2 + seq_len(p), drop = FALSE]
  h1_reach <- rowSums(w1) > 0
  h2_to_out <- as.logical(w3_h[1, ])
  h1_to_out <- if (any(h2_to_out)) colSums(w2_h[h2_to_out, , drop = FALSE]) > 0 else rep(FALSE, h1)
  h2_reach <- (if (any(h1_reach)) rowSums(w2_h[, h1_reach, drop = FALSE]) > 0 else rep(FALSE, h2)) | rowSums(w2_x) > 0
  a1 <- w1 & matrix(h1_to_out, nrow = h1, ncol = p)
  a2_h <- w2_h & outer(h2_to_out, h1_reach, "&")
  a2_x <- w2_x & matrix(h2_to_out, nrow = h2, ncol = p)
  a3_h <- w3_h & matrix(h2_reach, nrow = 1, ncol = h2)
  a3_x <- w3_x
  a2 <- cbind(a2_h, a2_x)
  a3 <- cbind(a3_h, a3_x)
  selected <- colSums(a1) > 0 | colSums(a2_x) > 0 | as.logical(a3_x[1, ])
  raw_retained <- sum(w1) + sum(w2) + sum(w3)
  candidate <- length(w1) + length(w2) + length(w3)
  retained <- sum(a1) + sum(a2) + sum(a3)
  list(m1 = a1, m2 = a2, m3 = a3, selected = selected,
       retained = retained, candidate = candidate,
       dparam = retained / candidate, native_density = raw_retained / candidate)
}

apply_ann_masks <- function(model, masks, device) {
  with_no_grad({
    model$fc1$weight$mul_(torch_tensor(masks$m1 * 1, dtype = torch_float(), device = device))
    model$fc2$weight$mul_(torch_tensor(masks$m2 * 1, dtype = torch_float(), device = device))
    model$out$weight$mul_(torch_tensor(masks$m3 * 1, dtype = torch_float(), device = device))
  })
}

predict_ann <- function(model, loader, device) {
  model$eval(); pred <- c()
  with_no_grad({
    coro::loop(for (b in loader) {
      x <- b[[1]]$to(device = device)
      pred <- c(pred, as.numeric(model(x)$detach()$cpu()))
    })
  })
  pred
}

run_is_ann_l1 <- function(dat, opt, device) {
  p <- ncol(dat$Xtr)
  dl <- make_loaders(dat, opt$batch, device)
  torch_manual_seed(opt$seed)
  model <- ISANNL1(p, opt$h1, opt$h2)$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = opt$ann_lr)
  t0 <- proc.time()[[3]]
  for (epoch in seq_len(opt$ann_epochs)) {
    model$train()
    coro::loop(for (b in dl$train) {
      x <- b[[1]]$to(device = device)
      y <- b[[2]]$to(device = device)
      optimizer$zero_grad()
      pred <- model(x)$view_as(y)
      loss <- nnf_mse_loss(pred, y) + opt$ann_lambda * ann_l1_penalty(model)
      loss$backward(); optimizer$step()
    })
    if (opt$verbose && (epoch == 1L || epoch %% 100L == 0L || epoch == opt$ann_epochs)) {
      message(sprintf("IS-ANN-L1 epoch=%d", epoch))
    }
  }
  runtime <- proc.time()[[3]] - t0
  masks <- ann_masks_active_paths(model, opt$ann_threshold)
  apply_ann_masks(model, masks, device)
  pred <- predict_ann(model, dl$test, device)
  list(
    pred = pred, selected = as.logical(masks$selected),
    retained = masks$retained, candidate = masks$candidate,
    dparam = masks$dparam, native_density = masks$native_density, runtime = runtime,
    rule = sprintf("Input-skip ANN with L1 penalty lambda=%g; |w|>%g then remove weights outside complete active paths", opt$ann_lambda, opt$ann_threshold)
  )
}

finish_row <- function(method, fit, dat, data_seed, fit_seed) {
  pm <- pred_metrics(fit$pred, dat$signal, dat$yte)
  sm <- selection_metrics(fit$selected, dat$truth)
  data.frame(
    method = method, status = "OK", seed = data_seed,
    n = nrow(dat$Xtr) + nrow(dat$Xte), p = ncol(dat$Xtr), n_active = sum(dat$truth),
    mse_signal = pm[["mse_signal"]], r2_signal = pm[["r2_signal"]],
    mse_y = pm[["mse_y"]], rmse_y = pm[["rmse_y"]],
    tpr = sm[["tpr"]], fpr = sm[["fpr"]], accuracy = sm[["accuracy"]],
    selected_support = sm[["selected_support"]],
    retained_weights = fit$retained, candidate_weights = fit$candidate,
    dparam = fit$dparam, d_edge = NA_real_, d_path = NA_real_,
    native_density = fit$native_density, runtime_sec = fit$runtime,
    native_rule = fit$rule, note = sprintf("fit_seed=%d", fit_seed),
    check.names = FALSE
  )
}

main <- function() {
  opt <- parse_args()
  if (is.null(opt$data_dir)) stop("Pass --data-dir PATH containing exported Saha CSV files")
  device <- resolve_bench_device(opt$device)
  dat <- load_data(opt$data_dir)
  methods <- trimws(strsplit(tolower(opt$methods), ",", fixed = TRUE)[[1]])
  allowed <- c("lbbnn_lrt", "lbbnn_flow", "islab_flow", "is_ann_l1")
  bad <- setdiff(methods, allowed)
  if (length(bad)) stop("Unknown methods: ", paste(bad, collapse = ", "))
  message(sprintf("Saha benchmark R methods | n=%d p=%d active=%d | architecture=%d->%d->%d->1 | device=%s",
                  nrow(dat$Xtr) + nrow(dat$Xte), ncol(dat$Xtr), sum(dat$truth), ncol(dat$Xtr), opt$h1, opt$h2, device))
  rows <- list()
  for (m in methods) {
    label <- switch(m, lbbnn_lrt = "LBBNN-LRT", lbbnn_flow = "LBBNN-FLOW",
                    islab_flow = "ISLaB-FLOW", is_ann_l1 = "IS-ANN-L1")
    message("\n[", label, "]")
    fit <- if (m == "is_ann_l1") run_is_ann_l1(dat, opt, device) else run_lbbnn(m, dat, opt, device)
    data_seed <- if (!is.null(dat$meta$seed) && !is.na(dat$meta$seed)) as.integer(dat$meta$seed) else NA_integer_
    row <- finish_row(label, fit, dat, data_seed, opt$seed)
    rows[[length(rows) + 1L]] <- row
    message(sprintf("MSE(signal)=%.4g R2=%.4f TPR=%.3f FPR=%.3f Acc=%.3f support=%d Dparam=%.4f time=%.1fs",
                    row$mse_signal, row$r2_signal, row$tpr, row$fpr, row$accuracy,
                    row$selected_support, row$dparam, row$runtime_sec))
  }
  ans <- do.call(rbind, rows)
  write.csv(ans, opt$out, row.names = FALSE)
  cat("\nFinal R-method benchmark\n")
  print(ans[, c("method", "mse_signal", "r2_signal", "tpr", "fpr", "accuracy", "selected_support", "dparam", "runtime_sec")], row.names = FALSE)
  cat("\nSaved:", normalizePath(opt$out, mustWork = FALSE), "\n")
}

main()
