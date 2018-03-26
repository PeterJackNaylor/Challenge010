nextflow run trainUNet.nf -c ../utils/nextflow.config --test_set ../../dataset/stage1_small_test/*/images/*.png \
                          --input_f ../../intermediary_files/Data/UNetData/small_data_unet --retrain 1\
                           -profile local -resume