NXF_WORK=/data/tmp/pnaylor/Challenge010 nextflow run trainUNet.nf -c ../utils/nextflow.config -profile GPU_kep \
                          --epoch 100 --thalassa 1 --real 1 --name UNetHistogramTW1 -resume --train Dist/DistTrain.py --validation Dist/DistValidate.py \
                          --test Dist/DistTest.py --input_f ../../intermediary_files/Data/HistoNormDist/data_dist_histonorm --test_set "../../intermediary_files/Data/HistoNorm/TestSet/data_unet_histonorm_test/*/images/*.png"
