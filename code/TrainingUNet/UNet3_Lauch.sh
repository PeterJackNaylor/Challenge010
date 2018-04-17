NXF_WORK=/data/tmp/pnaylor/Challenge010 nextflow run trainUNet.nf -c ../utils/nextflow.config -profile GPU_kep --train UNet3/UNet3.py --test UNet3/UNet3Test.py --validation UNet3/UNet3Validation.py \
                          --epoch 75 --thalassa 0 --real 1 --name UNet3 -resume --tfrecord_py TFRecord.py --full_train 1\
                          --input_f ../../intermediary_files/Data/UNet3/data_unet3 --test_set "../../intermediary_files/Data/HistoNorm/TestSet/data_unet_histonorm_test/*/images/*.png"
