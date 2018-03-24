nextflow run trainUNet.nf -c ../utils/nextflow.config -profile mines \
                          --epoch 20 --thalassa 1 --real 1 --name ContrastUNet -resume
