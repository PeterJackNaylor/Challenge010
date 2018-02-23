HOME=/share/data40T_v2/Peter nextflow run trainUNet.nf -c ../utils/nextflow.config \
                          --epoch 100 --thalassa 1 \
                          --real 1 --name ContrastUNet \
                          --profiles mines -resume
