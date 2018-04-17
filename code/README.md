This is our repo for the DSB 2018 Challenge by Thomas and Peter.

It is for the most a self contained code for creating data, training and final testing. However final ensembling methods are not part of the pipeline due to time so it is mannual from there.

As the weights can be automaticaly generated no weights are provided. We do in total 4 models. (1GB of weights... not fun to upload).

#To generate the data
Put the data you wish to train on in a dataset/stage1_train and the data you wish to test on in dataset/stage1_test. 
run the command "bash NextFlowPrepData.sh" in code/data, the data will be created in the intermediary/data folder.

External data such can be run by adding the external data in external_dataset and you can run the command "bash External_data from the code/data folder.
External data i used was Coelho2009_ISBI_NuclearSegmentation, Neeraj Kumar histopathology dataset and ourown dataset. TNBC Breast cancer patients.
#Run Training. 
Due to our different configurations (we have an SGE cluster and normal GPU computers) we have bash commands that our system specific. However, I would check the sh files in TrainingUNet that have "Kepler" in the name. (this would be the normal GPU computer). 
We launch ExternalData_UNetNormalize.sh, Dist_Kepler.sh, UNet3_Lauch.sh, Dist_HistogramTW_kep.sh. 

This will result in files called in intermediary folder. We show an example in result of how you can reorganise these several folders. You could stop here and submit (actually, our best results stops here). 
Optionnaly, we could do better because we still can do some post-processing (and in order to be non biais) we produce a training set for training the post-processing and a full training model over the entire dataset for when we actually want to predict. 
We wish to learn a final post-processing model that classifies if or not an edge is good.



#Train to know if an edge is good.
To train this model you can run: python EdgeClassification.py --nn_names  UNetDistHistogramTW2,UNetHistogramTW2,rcnn_external,UNet3,UNetExternal --type_agr UNetExternal --feature_extraction --train --predict_train .
You might need options such as --input, --output --modelfolder and --showfolder that are explained in that python file.

# To predict the final model you have to use the 
python EdgeClassification.py --input /Users/naylorpeter/Desktop/NucleiKaggle/results/FullModels --nn_names  UNetDistHistogramTW2,UNetHistogramTW2,rcnn_external,UNet3,UNetExternal --type_agr UNetExternal --predict_test

This is in order to use the full trained models with the unbias classifier edge detector. 

# ReCompute the csv outputfile

launch python compute_csvscore.py --input ../../results/edge_singleextra/output/Test/Test --output ../../results/edge_singleextra/output/Test/corected.csv from the Ensemble folder. 


PS: one model is missing in this, but from what we understand we are allowed to keep running a model if it doesn't mean touching the code anymore. ExternalData trained on the whole dataset is still ongoing and should arrive soon. 
