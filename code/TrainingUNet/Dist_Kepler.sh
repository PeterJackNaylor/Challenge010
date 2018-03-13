NXF_WORK=/data/tmp/pnaylor/Challenge010 nextflow run trainUNet.nf -c ../utils/nextflow.config \
                          --epoch 100 --train Dist/DistTrain.py --validation Dist/DistValidate.py \
                          --test Dist/DistTest.py--name Distance --real 1 --input_f ../../intermediary_files/Data/UNetData/data_dist \
                          -profile GPU_kep -resume

