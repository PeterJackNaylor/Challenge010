nextflow run trainUNet.nf -c ../utils/nextflow.config -profile mines \
                          --epoch 100 --thalassa 0 --real 1 --name DistHistogramTW1 -resume \
                          --input_f ../../intermediary_files/Data/HistoNorm/data_unet_histonorm --test_set "../../intermediary_files/Data/HistoNorm/TestSet/data_unet_histonorm_test/*/images/*.png"
