nextflow run trainUNet.nf -c ../utils/nextflow.config \
                          -epoch 100 \
                          -real 1 \
                          --profiles GPU_kep -resume