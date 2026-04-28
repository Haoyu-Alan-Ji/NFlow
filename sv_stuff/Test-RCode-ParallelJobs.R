
s = as.integer(commandArgs()[3])  
f = paste0("/projects/standard/ventz001/ventz001/F01-Current/Test-code/Test-Cluster-jobid-",s,".RData")
print(f)


X = matrix(s,5,5)
    save(X, file=f)
