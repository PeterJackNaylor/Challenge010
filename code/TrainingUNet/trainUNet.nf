#!/usr/bin/env nextflow

params.input_f = "../../intermediary_files/Data/UNetData/data_unet"
params.epoch = 1
params.model = "UNet.py"
params.size = 212
params.unet_like = 1
params.name = "UNet"
params.real = 0


FOLDS_PATH_GLOB = params.input_f + "/Slide_*"
INPUT_F = file(params.input_f)
FOLDS_POSSIBLE = file(FOLDS_PATH_GLOB, type: 'dir', followLinks: true)
NUMBER_OF_FOLDS = FOLDS_POSSIBLE.size() - 1
MODEL = file(params.model)
MINI_EPOCH = 1

// mini epoch is to take advantage of my img deformation and of a cluster
// unet_like is just in case i need to tweet the CreateRecord

TFRECORDS = file('TFRecord.py')

process CreateRecords {
    input:
    file py from TFRECORDS
    file path from INPUT_F
    val epoch from MINI_EPOCH
    each test from 0..NUMBER_OF_FOLDS

    output:
    set val("$test"), file("${params.name}.tfrecords") into TrainRecords
    """
    python $py --tf_record ${params.name}.tfrecords --path $path \\
               --test $test --size_train ${params.size} --unet ${params.unet_like} \\
               --seed 42 --split train
    """
}

COMPUTE_MEAN = file("ComputeMean.py")

process Meanfile {
    input:
    file py from COMPUTE_MEAN
    file path from INPUT_F
    output:
    file "mean_file.npy" into MEAN_ARRAY
    """
    python $py --input $path --output mean_file.npy
    """
}

if( params.real == 1 ) {
    LEARNING_RATE = [0.01, 0.001, 0.0001, 0.00001]
    WEIGHT_DECAY = [0.0005, 0.00005]
    N_FEATURES = [16, 32, 64]
    BATCH_SIZE = 16
}
else {
    LEARNING_RATE = [0.01, 0.001]
    WEIGHT_DECAY = [0.0005]
    N_FEATURES = [16]
    BATCH_SIZE = 16
}

process TrainModel {
    publishDir "../../intermediary_files/Training/${params.name}", overwrite:true
    if( params.real == 1 ) {
        beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
        afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
        maxfork 2
    }
    input:
    file py from MODEL
    file path from INPUT_F
    set test, file(rec) from TrainRecords
    file mean_array from MEAN_ARRAY
    val bs from BATCH_SIZE
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY
    each nfeat from N_FEATURES
    output:
    file "${params.name}__${lr}__${wd}__${nfeat}__fold-${test}" into LOG_FOLDER
    file "${params.name}__${lr}__${wd}__${nfeat}__fold-${test}.csv" into CSV_FOLDER

    """
    python $py --tf_record $rec --path $path --size_train ${params.size} --mean_file $mean_array \\
               --log ${params.name}__${lr}__${wd}__${nfeat}__fold-${test} --split train --epoch ${params.epoch} \\
               --batch_size $bs --learning_rate $lr --weight_decay $wd --n_features $nfeat --test $test
    """

}