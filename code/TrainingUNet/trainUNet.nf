#!/usr/bin/env nextflow

params.input_f = "../../intermediary_files/Data/UNetData/data_unet"
params.epoch = 1
params.train = "UNet/UNet.py"
params.validation = "UNet/UNetValidation.py"
params.test = "UNet/UNetTest.py"
params.retrain_py = "UNet/UNetRetrain.py"
params.size = 212
params.unet_like = 1
params.name = "UNet"
params.real = 0
params.test_set = "../../dataset/stage1_test/*/images/*.png"
params.thalassa = 0
params.info_pc = "../../intermediary_files/Data/train_test.csv"
params.retrain = 0

INFO_TAB = file(params.info_pc)
FOLDS_PATH_GLOB = params.input_f + "/Slide_*"
INPUT_F = file(params.input_f)
FOLDS_POSSIBLE = file(FOLDS_PATH_GLOB, type: 'dir', followLinks: true)
NUMBER_OF_FOLDS = FOLDS_POSSIBLE.size() - 1
MODEL_TRAIN = file(params.train)
MODEL_RETRAIN = file(params.retrain_py)
MODEL_TEST  = file(params.test)
MODEL_VALID = file(params.validation)
MINI_EPOCH = 1
INPUT_TEST = Channel.from(file(params.test_set))
PICK_MODEL = file('PickModel.py')
TFRECORDS = file('TFRecord.py')
COMPUTE_MEAN = file("ComputeMean.py")

// mini epoch is to take advantage of my img deformation and of a cluster
// unet_like is just in case i need to tweet the CreateRecord


process CreateRecords {
    clusterOptions "-S /bin/bash"
    input:
    file py from TFRECORDS
    file path from INPUT_F
    val epoch from MINI_EPOCH
    each test from 0..NUMBER_OF_FOLDS

    output:
    set val("$test"), file("${params.name}.tfrecords") into TrainRecords
    script:
    if( params.thalassa == 0 ){
        """
        python $py --tf_record ${params.name}.tfrecords --path $path \\
                   --test $test --size_train ${params.size} --unet ${params.unet_like} \\
                   --seed 42 --split train
        """
    } else {
        """
        PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
        function pyglib {
            /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
        }
        pyglib $py --tf_record ${params.name}.tfrecords --path $path \\
                   --test $test --size_train ${params.size} --unet ${params.unet_like} \\
                   --seed 42 --split train
        """
    }
}


process Meanfile {
    clusterOptions "-S /bin/bash"

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
    LEARNING_RATE = [0.001, 0.0001]
    WEIGHT_DECAY = [0.00005]
    N_FEATURES = [32]
    BATCH_SIZE = 8
}
else {
    LEARNING_RATE = [0.01]
    WEIGHT_DECAY = [0.0005]
    N_FEATURES = [16]
    BATCH_SIZE = 1
}

process TrainModel {
    clusterOptions "-S /bin/bash"
    publishDir "../../intermediary_files/Training/${params.name}", overwrite:true
    
    if( params.real == 1 ) {
        beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
        afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
    }
    if( params.thalassa == 1 ){
        queue "cuda.q"
        maxForks 2    
    } else {
        maxForks 1
    }

    input:
    file py from MODEL_TRAIN
    file path from INPUT_F
    set test, file(rec) from TrainRecords
    file mean_array from MEAN_ARRAY
    val bs from BATCH_SIZE
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY
    each nfeat from N_FEATURES
    output:
    set val("${params.name}__${lr}__${wd}__${nfeat}"), file("${params.name}__${lr}__${wd}__${nfeat}__fold-${test}") into LOG_FOLDER
    set val("${params.name}__${lr}__${wd}__${nfeat}"), file("${params.name}__${lr}__${wd}__${nfeat}__fold-${test}.csv") into CSV_FOLDER, CSV_FOLDER2

    script:
    if( params.thalassa == 0 ){
        """
        python $py --tf_record $rec --path $path --size_train ${params.size} --mean_file $mean_array \\
                   --log ${params.name}__${lr}__${wd}__${nfeat}__fold-${test} --split train --epoch ${params.epoch} \\
                   --batch_size $bs --learning_rate $lr --weight_decay $wd --n_features $nfeat --test $test
        """
    }
    else {
        """
        function pyglib {
            /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
        }
        pyglib $py --tf_record $rec --path $path --size_train ${params.size} --mean_file $mean_array \\
                   --log ${params.name}__${lr}__${wd}__${nfeat}__fold-${test} --split train --epoch ${params.epoch} \\
                   --batch_size $bs --learning_rate $lr --weight_decay $wd --n_features $nfeat --test $test
        """
    }
}


process PickBestModel {
    clusterOptions "-S /bin/bash"
    publishDir "../../intermediary_files/Training/${params.name}", pattern: "*.csv", overwrite:true

    input:
    file _ from CSV_FOLDER .collect()
    file py from PICK_MODEL
    output:
    file "${params.name}__*" into NAME, NAME_, NAME__
    file "test_tables.csv"
    """
    python $py
    """
}

NAME.map{it -> [it.name, "blank"]}.into{BEST_T;BEST_T2}
BEST_T.cross(LOG_FOLDER).map{it -> it[1][1]}.into{BEST_G_LOG; BEST_LOG_2}
BEST_T2.cross(CSV_FOLDER2).map{it -> it[1][1]}.set{BEST_G_CSV}


process FindingP1P2 {
    clusterOptions "-S /bin/bash"
    publishDir "../../intermediary_files/Training/${params.name}/Final", overwrite:true
    if( params.real == 1 ) {
        beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
        afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
        maxForks 2
    }
    if( params.thalassa == 1 ){
        queue "cuda.q"
    }
    input:
    file name from NAME_
    file(log) from BEST_G_LOG .collect()
    file(csv) from BEST_G_CSV .collect() // Not needed, because of fix in train symlink
    file path from INPUT_F
    file py from MODEL_VALID
    file mean_array from MEAN_ARRAY

    output:
    file "Hyper_parameter_selection.csv" into HP_SCORE, HP_SCORE2
    file "${name}__onTrainingSet"
    file "__summary_per_image.csv" into SUMMARY_TRAIN
    script:
    if( params.thalassa == 0 ){
        """
        python $py --path $path --mean_file $mean_array --name $name\\
                   --output ${name}__onTrainingSet
        """
    } else {
        """
        function pyglib {
            /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
        }
        pyglib $py --path $path --mean_file $mean_array --name $name\\
                   --output ${name}__onTrainingSet
        """
    }
}

if( params.retrain == 1 ) {
    process ReTraining {
        clusterOptions "-S /bin/bash"
        publishDir "../../intermediary_files/Training_stage_two/${params.name}/Final", overwrite:true
        if( params.real == 1 ) {
            beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
            afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
        }
        if( params.thalassa == 1 ){
            queue "cuda.q"
            maxForks 2    
        } else {
            maxForks 2
        }
        input:
        file name from NAME_
        file logilog from BEST_LOG_2
        file path from INPUT_F
        file mean_array from MEAN_ARRAY
        file sum from SUMMARY_TRAIN
        val bs from BATCH_SIZE
        file py from MODEL_RETRAIN
        file tab from INFO_TAB
        file hp from HP_SCORE2

        output:
        file "${logilog}" into BEST_LOG_FINAL
        file "retraining.csv"

        script:
        if( params.thalassa == 0 ){
            """
            cp -r ${logilog} retrain_${logilog}
            python ${py} --path $path --size_train ${params.size} --mean_file $mean_array \\
                       --log retrain_${logilog} --split train --epoch ${params.epoch} \\
                       --batch_size $bs --table $sum --info $tab --hp $hp
            """
        }
        else {
            """
            cp -r ${logilog} retrain_${logilog}
            function pyglib {
                /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
            }
            pyglib ${py} --path $path --size_train ${params.size} --mean_file $mean_array \\
                       --log retrain_${logilog} --split train --epoch ${params.epoch} \\
                       --batch_size $bs --table $sum --info $tab --hp $hp
            """
        }
    }
}
else
{
     BEST_LOG_FINAL = BEST_LOG_2
}



process PredictTestSet {
    clusterOptions "-S /bin/bash"

    publishDir "../../intermediary_files/TestSet/${params.name}", overwrite:true
    if( params.real == 1 ) {
        beforeScript "source \$HOME/CUDA_LOCK/.whichNODE"
        afterScript "source \$HOME/CUDA_LOCK/.freeNODE"
        maxForks 2
    }
    input:
    file py from MODEL_TEST
    file __ from INPUT_TEST .collect()
    file mean_array from MEAN_ARRAY
    file _ from BEST_LOG_FINAL .collect()
    file name from NAME__
    file hp from HP_SCORE
    output:
    file "${params.name}_sampleTest"
    file "${params.name}_PredFile.csv"
    script:
    if( params.thalassa == 0 ){
        """
        python $py --mean_file $mean_array --hp $hp \\
                   --name $name --output_csv ${params.name}_PredFile.csv \\
                   --output_sample ${params.name}_sampleTest
        """
    } else {
        """
        PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
        function pyglib {
            /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
        }
        pyglib $py --mean_file $mean_array --hp $hp \\
                   --name $name --output_csv ${params.name}_PredFile.csv \\
                   --output_sample ${params.name}_sampleTest
        """
    }

}
