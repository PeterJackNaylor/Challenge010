nextflow run trainUNet.nf -c ../utils/nextflow.config -profile mines \
                          --epoch 100 --thalassa 1 --real 1 --name ReTrainUNet --retrain 1 -resume
