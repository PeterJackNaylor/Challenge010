nextflow run trainUNet.nf -c ../utils/nextflow.config -profile mines --full_train 1\
                          --epoch 150 --thalassa 1 --real 1 --name UNetHistogramTW2All -resume \
                          --input_f ../../intermediary_files/Data/HistoNorm/data_unet_histonorm --test_set "../../intermediary_files/Data/HistoNorm/TestSet/data_unet_histonorm_test/*/images/*.png"
