rm -r -f ./model/$1
mkdir ./model/$1
scp petschackm@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2163/sts-analysis/model/$1/model.safetensors ./model/$1/model.safetensors