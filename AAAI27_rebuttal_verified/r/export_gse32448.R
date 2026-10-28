args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript export_gse32448.R output_folder")
if (!requireNamespace("GEOquery", quietly = TRUE)) stop("Install GEOquery via BiocManager::install('GEOquery')")
if (!requireNamespace("Biobase", quietly = TRUE)) stop("Install Biobase via BiocManager::install('Biobase')")
dir.create(args[1], recursive = TRUE, showWarnings = FALSE)
# No need to download the much larger raw CEL archive or platform annotation.
geo_args <- list(GEO = "GSE32448", GSEMatrix = TRUE,
                 getGPL = FALSE, AnnotGPL = FALSE)
if ("returnType" %in% names(formals(GEOquery::getGEO))) {
    geo_args$returnType <- "ExpressionSet"
}
gse <- do.call(GEOquery::getGEO, geo_args)
if (length(gse) > 1) {
    ii <- which(vapply(gse, function(x) Biobase::annotation(x) == "GPL570", logical(1)))
    if (!length(ii)) stop("GPL570 not found")
    eset <- gse[[ii[1]]]
} else eset <- gse[[1]]
mat <- as.data.frame(t(Biobase::exprs(eset)))
mat <- cbind(sample_id = rownames(mat), mat)
metadata <- Biobase::pData(eset)
metadata$sample_id <- rownames(metadata)
if (!"title" %in% names(metadata)) stop("GEO phenotype lacks sample title")
write.csv(mat, file.path(args[1], "expression.csv"), row.names = FALSE)
write.csv(metadata, file.path(args[1], "metadata.csv"), row.names = FALSE)
cat("[export] GSE32448:", nrow(mat), "samples;", ncol(mat)-1, "probes\n")
