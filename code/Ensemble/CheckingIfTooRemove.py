from sklearn.ensemble import RandomForestClassifier
import skimage
from skimage.io import imread, imsave
import numpy as np
from GatheringFiles import GatherMultipleModels, Model_gen

def get_image_features(nn_label_img, nn_probmap):
    
    res = {}
    id_lab = -1
    for nn_lab_name in nn_label_img.keys():
        id_lab += 1
        label_img = nn_label_img[nn_lab_name]
        
        for nn_name in sorted(nn_probmap):
            im_prob = nn_probmap[nn_name]
            props = skimage.measure.regionprops(label_img, im_prob)
            
            id_obj = range(1, len(props) + 1)
            id_lab_vec = np.ones(len(props)) * id_lab

            for feature_name, vec in zip(['id_obj', 'id_lab_vec'], [id_obj, id_lab_vec]):
                if not feature_name in res:
                    res[feature_name] = vec
                else:
                    res[feature_name] = np.concatenate([res[feature_name], vec])

            intensity = np.array([obj.mean_intensity for obj in props])
            moment20 = np.array([obj.weighted_moments_normalized[2,0] for obj in props])
            moment02 = np.array([obj.weighted_moments_normalized[0,2] for obj in props])
            moment11 = np.array([obj.weighted_moments_normalized[1,1] for obj in props])

            for feature_name, vec in zip(['intensity_%s', 'intensity_moment20_%s', 'intensity_moment02_%s', 'intensity_moment11_%s'],
                                         [intensity, moment20, moment02, moment11]):
                feature_name_loaded = feature_name % nn_name
                if not feature_name_loaded in res:
                    res[feature_name_loaded] = vec
                else:
                    res[feature_name_loaded] = np.concatenate([res[feature_name_loaded], vec])

            solidity = np.array([obj.solidity for obj in props])    
            area = np.array([obj.area for obj in props])
        
        
            eigenvalue_1 = np.array([.5 * (obj.moments_central[0,2] + obj.moments_central[2,0]) + 
                                     .5 * np.sqrt(4 * obj.moments_central[1,1]**2 + (obj.moments_central[0,2] - obj.moments_central[2,0])**2)
                                     for obj in props])
            eigenvalue_2 = np.array([.5 * (obj.moments_central[0,2] + obj.moments_central[2,0]) - 
                                     .5 * np.sqrt(4 * obj.moments_central[1,1]**2 + (obj.moments_central[0,2] - obj.moments_central[2,0])**2)
                                     for obj in props])
            ellipse_area = np.pi * 4.0 * np.sqrt(eigenvalue_1 * eigenvalue_2) / area.astype(np.float)

            ellipse_feature = np.abs(area.astype(np.float) - ellipse_area) / np.clip(ellipse_area, a_min=1.0, a_max=None)
            
            for feature_name, vec in zip(['solidity', 'area', 'ellipse_feature'],
                                         [solidity, area, ellipse_feature]):
                if not feature_name in res:
                    res[feature_name] = vec
                else:
                    res[feature_name] = np.concatenate([res[feature_name], vec])

    return res


def get_tp_and_fp(img_ground_truth, img_prediction_label, coverage = 0.5):
    img_ground_truth_bin = img_ground_truth > 0
    props = skimage.measure.regionprops(img_prediction_label, img_ground_truth_bin)            
    intensity = np.array([obj.mean_intensity for obj in props])
    return (intensity > coverage)

class RF_nodes(object):
    def __init__(self, nn_names):
        self.rf = RandomForestClassifier(n_estimators=500, oob_score=True, class_weight="balanced")
        self.features = [
                         'area',
                         'solidity',
                         'ellipse_feature',
#                         'intensity_moment02_ContrastUNet', 
#                         'intensity_moment20_ContrastUNet',
#                         'intensity_moment11_ContrastUNet',
#                         'intensity_moment02_Dist',
#                         'intensity_moment20_Dist',
#                         'intensity_moment11_Dist'
                       ] + ["intensity_" + name for name in nn_names]
        self.nn_names = nn_names
        self.coverage = 0.7

    def get_tp_and_fp(self, img_ground_truth, img_prediction_label):
        img_ground_truth_bin = img_ground_truth > 0
        props = skimage.measure.regionprops(img_prediction_label, img_ground_truth_bin)            
        intensity = np.array([obj.mean_intensity for obj in props])
        return (intensity > self.coverage)

    def get_training_dict(self):
        res = {}   
        nn_names = self.nn_names
        print 'collecting data for ', nn_names
        general_dic = GatherMultipleModels(nn_names, "Train")
        for image_name, dic in Model_gen(general_dic, tags=["LabeledBin", "output_DNN", "bin"]):
            nn_probmap = dic["output_DNN"]
            nn_label_img = dic["LabeledBin"]
            gt = skimage.measure.label(dic["bin"][nn_names[0]])
            yvec = np.array([], dtype=np.uint8)
            for mod in nn_names:
                pred_lbl = nn_label_img[mod]
                yvec = np.concatenate([yvec, self.get_tp_and_fp(gt, pred_lbl)])

            res[image_name] = get_image_features(nn_label_img, nn_probmap)
            res[image_name]['y'] = yvec
            print 'image %s : %i / %i' % (image_name, sum(yvec), len(yvec))
        return res
    def make_design_matrix(self, training_dict, image_list=None):
        if image_list is None:
            image_list = sorted(training_dict.keys())
        X = np.vstack([np.hstack([training_dict[image][feature] for image in image_list]) for feature in self.features]).T
        yvec = np.hstack([training_dict[image]['y'] for image in image_list])
        return X, yvec

    def get_training_set(self, SpecificNNname):
        training_dict = self.get_training_dict()
        X, yvec = self.make_design_matrix(training_dict)
        for i, nn in enumerate(sorted(self.nn_names)):
            if nn == SpecificNNname:
                break
        indices = training_dict["id_lab_vec"] == i
        X = X[indices]
        yvec = yvec[indices]
        return X, yvec

    def train(self, X, yvec, save_model=True):
        self.rf.fit(X, yvec)

        # report:
        print 'training succeeded. OOB accuracy : %.2f' % self.rf.oob_score_
        print 'Number of samples: %i ' % X.shape[0]
        print 'Number of features: %i' % X.shape[1]
        print 'Number of positive samples: %i (%.2f)' % (np.sum(yvec), np.float(np.sum(yvec)) / len(yvec))
        print 'Number of negative samples: %i (%.2f)' % ((len(yvec) - np.sum(yvec)), (1.0 - np.float(np.sum(yvec)) / len(yvec)))

        # if save_model:
        #     training_set = {'X': X, 'y': yvec}
        #     fp = open(os.path.join(self.model_folder, 'training_set.pickle'), 'w')
        #     pickle.dump(training_set, fp)
        #     fp.close()

        #     fp = open(os.path.join(self.model_folder, 'rf.pickle'), 'w')
        #     pickle.dump(self.rf, fp)
        #     fp.close()
        return 

    def __call__(self, SpecificNNname):
        X, yvec = self.get_training_set(SpecificNNname)
        self.train(X, yvec, True)

    def test(self, X):
        print "not implemented"
def AddPictures(dic, tags=["LabeledBin", "TPandFP"]):
    models = dic.keys()
    file_names = dic[models[0]].keys()
    if "LabeledBin" in tags:
        for mod in models:
            for _ in file_names:
                name = dic[mod][_]["rgb"].replace("rgb", "LabeledBin")
                bin_color = skimage.measure.label(np.sum(imread(dic[mod][_]["colored_pred"]), axis=2))
                imsave(name, bin_color.astype('uint16'))

if __name__ == '__main__':
    create_label = False
    NN_NAMES = ["UNetHistogramTW2", "UNetDistHistogramTW2", "CNN3", "UNetExternal"]
    if create_label:
        general_dic = GatherMultipleModels(NN_NAMES, "Train")
        AddPictures(general_dic, tags="LabeledBin")
    model = RF_nodes(nn_names=NN_NAMES)
    X, yvec = model.get_training_set("UNetExternal")
    model.train(X, yvec, True)

