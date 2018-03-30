nextflow run trainUNet.nf -c ../utils/nextflow.config -profile GPU_kep --train UNet4/UNet.py --test UNet4/UNetTest.py --validation UNet4/UNetValidation.py \
                          --epoch 100 --thalassa 0 --real 1 --name UNet4 -resume --tfrecord_py UNet4/TFRecord4.py\
                          --input_f ../../intermediary_files/Data/fuse4/TrainSet/data_fuse4 --test_set "../../intermediary_files/Data/fuse4/TestSet/data_fuse4_test/*/images/*.png"
