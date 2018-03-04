NXF_WORK=/data/tmp/pnaylor/Challenge010 nextflow run trainUNet.nf -c ../utils/nextflow.config \
                          --epoch 100 --name UNetElast\
                          --real 1 \
                          -profile GPU_kep -resume
