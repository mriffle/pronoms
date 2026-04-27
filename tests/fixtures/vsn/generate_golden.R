#!/usr/bin/env Rscript
# Generate golden reference outputs from the R `vsn` package.
#
# Each case writes:
#   <name>_input.csv         input matrix (proteins x samples, the R orientation)
#   <name>_output.csv        normalized output (same orientation)
#   <name>_coef.csv          fitted (a_j, b_j) coefficients, one row per sample,
#                            in *log space* — i.e. the columns are
#                            (offset_a, log_factor_b) where the actual factor is
#                            exp(log_factor_b).
#   <name>_meta.csv          one-row CSV with: lts.quantile, sigsq, hoffset,
#                            mu (as a JSON-ish vector), n_features, n_samples.
#
# The Python tests load these and assert that our native engine produces
# matching outputs within a tight tolerance.

suppressPackageStartupMessages({
  library(vsn)
  library(Biobase)
})

OUT_DIR <- "."

# --- helper: run vsn and dump everything we'll want to compare against ----

dump_case <- function(name, X, lts_quantile = 0.9, minDataPointsPerStratum = 42L) {
  # X is proteins x samples (R convention).
  eset <- ExpressionSet(assayData = X)
  fit <- vsn2(
    eset,
    lts.quantile = lts_quantile,
    minDataPointsPerStratum = minDataPointsPerStratum,
    verbose = FALSE
  )
  Y <- exprs(fit)

  # Coefficients: array (nstrata, nsamples, 2). For unstratified default, nstrata=1.
  coef <- fit@coefficients
  stopifnot(length(dim(coef)) == 3, dim(coef)[1] == 1, dim(coef)[3] == 2)
  flat <- cbind(a = coef[1, , 1], b_log = coef[1, , 2])

  # Sigma^2 / mu / hoffset are stored on the vsn object too.
  sigsq <- fit@sigsq
  mu <- fit@mu
  # hoffset is computed inside predict() but we can mirror its formula from the
  # vsn2.R source (vsn2.R:365):  hoffset = log2(2 * exp(rowMeans(coef[,,2])))
  hoffset <- log2(2 * exp(rowMeans(coef[, , 2, drop = FALSE])))

  write.csv(X, file.path(OUT_DIR, paste0(name, "_input.csv")), row.names = FALSE)
  write.csv(Y, file.path(OUT_DIR, paste0(name, "_output.csv")), row.names = FALSE)
  write.csv(flat, file.path(OUT_DIR, paste0(name, "_coef.csv")), row.names = FALSE)
  meta <- data.frame(
    name = name,
    lts_quantile = lts_quantile,
    minDataPointsPerStratum = minDataPointsPerStratum,
    sigsq = sigsq,
    hoffset = hoffset[1],
    n_features = nrow(X),
    n_samples = ncol(X)
  )
  write.csv(meta, file.path(OUT_DIR, paste0(name, "_meta.csv")), row.names = FALSE)
  # Save mu separately because it's a long vector.
  write.csv(data.frame(mu = mu), file.path(OUT_DIR, paste0(name, "_mu.csv")), row.names = FALSE)

  cat(sprintf("[%s] features=%d samples=%d sigsq=%.6f hoffset=%.6f\n",
              name, nrow(X), ncol(X), sigsq, hoffset[1]))
  invisible(NULL)
}

# --- Case 1: small synthetic, 200 features x 6 samples, lognormal -----------

set.seed(1)
n_feat <- 200
n_samp <- 6
true_a <- runif(n_samp, -2, 2)
true_b <- runif(n_samp, 0.5, 2.0)
mu_true <- rlnorm(n_feat, meanlog = 5, sdlog = 1)
X1 <- outer(mu_true, rep(1, n_samp))
# Apply linear shift+scale per sample, then add gaussian noise.
for (j in seq_len(n_samp)) {
  X1[, j] <- X1[, j] / true_b[j] - true_a[j] / true_b[j] + rnorm(n_feat, 0, 0.2 * mu_true)
}
# Ensure positivity not required, vsn handles negatives.
dump_case("synth_small", X1, lts_quantile = 0.9)

# --- Case 2: medium synthetic, 1000 features x 8 samples --------------------

set.seed(7)
n_feat <- 1000
n_samp <- 8
mu_true <- rlnorm(n_feat, meanlog = 4, sdlog = 1.2)
X2 <- matrix(0, n_feat, n_samp)
for (j in seq_len(n_samp)) {
  scale <- runif(1, 0.3, 3.0)
  shift <- runif(1, -0.5, 0.5) * mean(mu_true)
  X2[, j] <- scale * mu_true + shift + rnorm(n_feat, 0, 0.15 * mu_true)
}
dump_case("synth_medium", X2, lts_quantile = 0.9)

# --- Case 3: kidney dataset, the canonical VSN example ----------------------

data("kidney", package = "vsn")
Xk <- exprs(kidney)
dump_case("kidney", Xk, lts_quantile = 0.9)

# --- Case 4: same small data but with a more aggressive lts.quantile -------

set.seed(1)
n_feat <- 200
n_samp <- 6
true_a <- runif(n_samp, -2, 2)
true_b <- runif(n_samp, 0.5, 2.0)
mu_true <- rlnorm(n_feat, meanlog = 5, sdlog = 1)
X4 <- outer(mu_true, rep(1, n_samp))
for (j in seq_len(n_samp)) {
  X4[, j] <- X4[, j] / true_b[j] - true_a[j] / true_b[j] + rnorm(n_feat, 0, 0.2 * mu_true)
}
# Add some outliers so LTS trimming has work to do.
X4[1, 1] <- X4[1, 1] * 100
X4[5, 3] <- X4[5, 3] * 50
dump_case("synth_lts50", X4, lts_quantile = 0.5)
dump_case("synth_lts99", X4, lts_quantile = 0.99)

# --- Case 5: tiny — check we still run on near-minimum sized strata --------

set.seed(2)
Xt <- matrix(rlnorm(50 * 4, meanlog = 5, sdlog = 1), 50, 4)
dump_case("synth_tiny", Xt, lts_quantile = 0.9, minDataPointsPerStratum = 10L)

cat("\nAll golden cases written.\n")
