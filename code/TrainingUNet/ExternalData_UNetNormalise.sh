nextflow run trainUNet.nf -c ../utils/nextflow.config -profile GPU_kep  \
                          --epoch 150 --thalassa 0 --real 1 --name UNet_ExternalData -resume --tfrecord_py TFRecord.py\
                          --input_f ../../intermediary_files/ExtraData/HistoNorm/data_unet_histonorm --test_set "../../intermediary_files/ExtraData/HistoNorm/TestSet/data_unet_histonorm_test/*/images/*.png"
