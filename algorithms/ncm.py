import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.spatial.distance import cdist
from utils import save_model

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        #w_norm = self.weight.data.norm(dim=1, keepdim=True)
        #w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        #x_norm = input.data.norm(dim=1, keepdim=True)
        #x_norm = x_norm.expand_as(input).add_(self.epsilon)
        #w = self.weight.div(w_norm)
        #x = input.div(x_norm)
        out = F.linear(F.normalize(input, p=2,dim=1), \
                       F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class SplitCosineLinear(nn.Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) #concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out


cur_features = []
ref_features = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global curr_features
    curr_features = inputs[0]


class NCM():
    def __init__(self, total_class, dictionary_size=1000, num_epoch=100, shot_memory_max=10, img_size=[32, 32, 3],
                 data_type="image_array"):
        self.dictionary_size = dictionary_size
        self.shot_memory_max = shot_memory_max
        self.total_class = total_class
        if data_type == "image_array":
            self.memory = np.zeros((total_class, shot_memory_max, *img_size))
        elif data_type == "image_path":
            self.memory = np.chararray((total_class, shot_memory_max), itemsize=600)
        self.num_epoch = num_epoch
        self.prev_session_numclass = 0
        self.data_type = data_type


    def train(self, args, model, trainset, testset, first_testset, session, curr_num_classes=60, incr_class_num=5):


        print("SESSION!:, ", str(session))
        #test_batch_size = 64
        start_iteration = 0
        lamda = 5.
        num_epoch = 160
        num_features = model.feature_size

        ref_model, curr_model,  lambda_mult = self.set_modelclassifier( args, model, session, start_iteration, num_increment=incr_class_num)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,   weight_decay=2e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)


        # if session > start_iteration:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        #     #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-4)
        #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        #     num_epoch = 50
        # else:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        #     #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,   weight_decay=2e-4)
        #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
        #     num_epoch = 160


        if session > start_iteration:
            ref_model = ref_model.cuda()
            ref_model.eval()
            curr_model = curr_model.cuda()

            shape = [-1, *self.memory.shape[2:]]
            num_sample_perclass = self.shot_memory_max#self.dictionary_size//(curr_num_classes-incr_class_num)
            additional_memory = self.memory[:curr_num_classes-incr_class_num, :num_sample_perclass].reshape(shape)
            # for jj in self.memory:
            #     print(jj)
            additional_labels = np.expand_dims(np.array([ i for i in range(curr_num_classes-incr_class_num)]), axis=1)#.repeat(self.dictionary_size)
            additional_labels = np.repeat(additional_labels, num_sample_perclass, axis=1).reshape(-1)
            X_train    = np.concatenate((trainset._x, additional_memory),axis=0)
            Y_train    = np.concatenate((trainset._y, additional_labels), axis=0)

        else:
            X_train    = trainset._x#np.concatenate((trainset.train_data, self.memory),axis=0)
            Y_train    = trainset._y#np.concatenate((trainset.train_labels, self.memory))

        if args.rs_ratio > 0:
            #1/rs_ratio = (len(X_train)+len(X_protoset)*scale_factor)/(len(X_protoset)*scale_factor)
            scale_factor = (len(trainset) * args.rs_ratio) / (len(self.memory) * (1 - args.rs_ratio))
            rs_sample_weights = np.concatenate((np.ones(len(trainset)), np.ones(len(self.memory))*scale_factor))
            #number of samples per epoch, undersample on the new classes
            #rs_num_samples = len(X_train) + len(X_protoset)
            rs_num_samples = int(len(trainset) / (1 - args.rs_ratio))
            print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(trainset), len(self.memory), rs_num_samples))


        trainset._x = X_train#.astype('uint8')
        trainset._y = Y_train
        if session > start_iteration and args.rs_ratio > 0 and scale_factor > 1:
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
                                                      shuffle=False, sampler=train_sampler, num_workers=2)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=2)




        if session > 0 and session < 21:
            for epoch in range(num_epoch): ## CHANGE
                #train
                curr_model.train()

                train_loss = 0
                train_loss1 = 0
                train_loss2 = 0
                correct = 0
                total = 0
                lr_scheduler.step()


                for batch_idx, (inputs, targets, task_id) in enumerate(trainloader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    optimizer.zero_grad()

                    if session == start_iteration:
                        outputs = curr_model(inputs)
                        loss = F.cross_entropy(outputs, targets)
                    else:
                        cur_features, outputs = curr_model(inputs, is_feature=True)
                        with torch.no_grad():
                            ref_features, _ = ref_model(inputs, is_feature=True)
                        loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                                                         torch.ones(inputs.shape[0]).cuda())*lamda#*lamda#.to(device)) * lamda
                        #print(targets)
                        loss2 = F.cross_entropy(outputs, targets)
                        loss = loss1 + loss2
                    loss.backward()
                    optimizer.step()
                    #break

                train_loss += loss.item()
                total += targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()



        save_model(model, 'session_'+str(session)+'.pth', './save/')

        mean_class = self.update_memory(model, trainset, curr_num_classes, session=session, num_features=num_features)
        self.prev_session_numclass = curr_num_classes


        print('Computing accuracy for not FORGET...')
        # evalset.test_data = testset#.astype('uint8')
        # evalset.test_labels = map_Y_valid_ori
        first_evalloader = torch.utils.data.DataLoader(first_testset, batch_size=128,
                                                 shuffle=False, num_workers=2)
        ori_acc = self.compute_accuracy(model, mean_class, first_evalloader)

        print('Computing accuracy on the original batch of classes...')

        evalloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                 shuffle=False, num_workers=2)
        ori_acc = self.compute_accuracy(model, mean_class, evalloader)


        return curr_model

    def compute_features(self, feature_model, loader, num_samples, num_features=512, device=None):
        if device is None:
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            feature_model.eval()

            features = None#np.zeros([num_samples, num_featurescurrent_selected_class_samples])
            start_idx = 0
            with torch.no_grad():
                for inputs, targets, task_id in loader:
                    inputs = inputs.cuda()#to(device)
                    if features is None:
                        features, _ = feature_model(inputs, is_feature=True)
                        features = features.cpu().numpy()
                    else:
                        features_new, _ = feature_model(inputs, is_feature=True)
                        features_new = features_new.cpu().numpy()
                        features = np.concatenate((features, features_new), axis=0)
                    #features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(feature_model(inputs))
                    start_idx = start_idx+inputs.shape[0]
            #assert(start_idx==num_samples)
            return features

    # def update_memory(self, model, trainset, num_classes, num_shot, session=1, num_features=512):
    #     class_means = np.zeros((num_features, new_classes, 2)) # 0 for icarl, 1 for NCM
    #
    #     input_data = trainset.train_data
    #     labels = trainset.train_labels
    #     num_selected_samples = self.dictionary_size//curr_num_classes
    #
    #     print('Updating exemplar set...')
    #     for curr_class in range(curr_num_classes):
    #         idx_to_class = [idx for idx in range(len(labels)) if curr_class == labels[idx]]#np.where(labels==curr_class)
    #         current_selected_class_samples = input_data[idx_to_class]
    #
    #         #for iter_dico in range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl):
    #         # Possible exemplars in the feature space and projected on the L2 sphere
    #         #evalset.test_data = prototypes[iter_dico].astype('uint8')
    #         #evalset.test_labels = np.zeros(evalset.test_data.shape[0]) #zero labels
    #         trainset.train_data = current_selected_class_samples#self.memory[curr_class]
    #         trainset.train_labels = np.zeros(trainset.train_data.shape[0]) #zero labels
    #         loader = torch.utils.data.DataLoader(trainset, batch_size=32,
    #                                              shuffle=False, num_workers=2)
    #         num_samples = trainset.train_data.shape[0]
    #         mapped_prototypes = self.compute_features(model, loader, num_samples, num_features=num_features)
    #         D = mapped_prototypes.T
    #         D = D/np.linalg.norm(D,axis=0)
    #
    #         # Herding procedure : ranking of the potential exemplars
    #         mu  = np.mean(D, axis=1)
    #         scores   = np.dot(mu, D)
    #
    #         idxs = scores.argsort()[-num_selected_samples:][::-1]
    #         self.memory[curr_class, :len(idxs)] = current_selected_class_samples[idxs]
    #
    #         class_means[:, curr_class, 0] = mu#np.dot(D,alph)#(np.dot(D,alph)+np.dot(D2,alph))/2
    #         class_means[:, curr_class, 0] /= np.linalg.norm(class_means[:, curr_class, 1])
    #
    #         #alph = np.ones(self.dictionary_size)/self.dictionary_size
    #         class_means[:, curr_class, 1] = mu#np.dot(D,alph)#(np.dot(D,alph)+np.dot(D2,alph))/2
    #         class_means[:, curr_class, 1] /= np.linalg.norm(class_means[:, curr_class, 1])
    #
    #     return class_means

    def update_memory(self, model, trainset, curr_num_classes, session=1, num_features=512):
        class_means = np.zeros((num_features, curr_num_classes, 2)) # 0 for icarl, 1 for NCM

        #if self.data_type == "image_array":
        input_data = np.copy(trainset._x)
        labels =  np.copy(trainset._y)
        # elif self.data_type == "image_path":
        #     loader = torch.utils.data.DataLoader(trainset, batch_size=64,
        #                                          shuffle=False, num_workers=2)
        #
        #     input_data = None
        #     labels = []
        #     for inputs, targets, task_id in loader:
        #         if input_data is None:
        #             input_data = inputs
        #         else:
        #             input_data = np.concatenate((input_data, inputs), axis=0)
        #         labels = np.concatenate((labels, targets), axis=0)


        ## CSIMON change
        num_selected_samples = self.shot_memory_max#self.dictionary_size//curr_num_classes

        print('Updating exemplar set...')
        for curr_class in range(curr_num_classes):
            idx_to_class = [idx for idx in range(len(labels)) if curr_class == labels[idx]]#np.where(labels==curr_class)
            current_selected_class_samples = input_data[idx_to_class]

            trainset._x = current_selected_class_samples#self.memory[curr_class]
            trainset._y = np.zeros(trainset._x.shape[0]) #zero labels
            # num_samples = trainset._x.shape[0]
            # mapped_prototypes = self.compute_features(model, loader, num_samples, num_features=num_features)
            # D = mapped_prototypes.T
            # D = D/np.linalg.norm(D,axis=0)
            #
            # # Herding procedure : ranking of the potential exemplars
            # mu  = np.mean(D, axis=1)
            # scores   = np.dot(mu, D)
            D, mu, scores = self.calculate_prototype_and_scores(model, trainset, num_features)


            idxs = scores.argsort()[-num_selected_samples:][::-1]

            #### CHANGE THIS TO CHANGE MEMORY FIT SIZE
            if curr_class >= self.prev_session_numclass:
                self.memory[curr_class, :len(idxs)] = current_selected_class_samples[idxs]
                #self.memory[curr_class] = current_selected_class_samples[idxs]

            class_means[:, curr_class, 0] = mu#np.dot(D,alph)#(np.dot(D,alph)+np.dot(D2,alph))/2
            class_means[:, curr_class, 0] /= np.linalg.norm(class_means[:, curr_class, 0])

            #alph = np.ones(self.dictionary_size)/self.dictionary_size
            class_means[:, curr_class, 1] = mu#np.dot(D,alph)#(np.dot(D,alph)+np.dot(D2,alph))/2
            class_means[:, curr_class, 1] /= np.linalg.norm(class_means[:, curr_class, 1])

        return class_means


    def calculate_prototype_and_scores(self, model, trainset, num_features):
        loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                             shuffle=False, num_workers=2)
        num_samples = trainset._x.shape[0]
        mapped_prototypes = self.compute_features(model, loader, num_samples, num_features=num_features)
        D = mapped_prototypes.T
        D = D/np.linalg.norm(D,axis=0)

        # Herding procedure : ranking of the potential exemplars
        mu  = np.mean(D, axis=1)
        scores   = np.dot(mu, D)

        return D, mu, scores


    def set_modelclassifier(self, args, model, session, start_iteration=0, num_increment=5):
        if session == start_iteration:
            in_features = model.fc.in_features
            out_features = model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            ref_model = None
            lamda_mult = None
        elif session == start_iteration+1:
            ref_model = copy.deepcopy(model)
            in_features = model.fc.in_features
            out_features = model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            new_fc = SplitCosineLinear(in_features, out_features, num_increment)
            new_fc.fc1.weight.data = model.fc.weight.data
            #new_fc.sigma.data = model.fc.sigma.data
            model.fc = new_fc.cuda()
            lamda_mult = out_features*1.0 / num_increment
        else:
            ref_model = copy.deepcopy(model)
            in_features = model.fc.in_features
            out_features1 = model.fc.fc1.out_features
            out_features2 = model.fc.fc2.out_features
            print("in_features:", in_features, "out_features1:", \
                  out_features1, "out_features2:", out_features2)
            new_fc = SplitCosineLinear(in_features, out_features1+out_features2, num_increment)
            new_fc.fc1.weight.data[:out_features1] = model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = model.fc.fc2.weight.data
            #new_fc.sigma.data = model.fc.sigma.data
            model.fc = new_fc.cuda()
            lamda_mult = (out_features1+out_features2)*1.0 / (num_increment)

        return ref_model, model, lamda_mult





    def compute_accuracy(self, model, class_means, evalloader, scale=None, print_info=True, device=None):
        # if device is None:
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()


        correct = 0
        correct_icarl = 0
        correct_ncm = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, task_id) in enumerate(evalloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                total += targets.size(0)

                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)
                if scale is not None:
                    assert(scale.shape[0] == 1)
                    assert(outputs.shape[1] == scale.shape[1])
                    outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
                _, predicted = outputs.max(1)

                correct += predicted.eq(targets).sum().item()

                outputs_feature, _ = model(inputs, is_feature=True)
                outputs_feature = outputs_feature.cpu().numpy()
                # Compute score for iCaRL
                sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
                score_icarl = torch.from_numpy((-sqd_icarl).T).cuda()#.to(device)
                _, predicted_icarl = score_icarl.max(1)
                correct_icarl += predicted_icarl.cuda().eq(targets).sum().item()
                # Compute score for NCM
                sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
                score_ncm = torch.from_numpy((-sqd_ncm).T).cuda()#.to(device)
                _, predicted_ncm = score_ncm.max(1)
                correct_ncm += predicted_ncm.cuda().eq(targets).sum().item()

        if print_info:
            print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))
            print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
            print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))

        cnn_acc = 100.*correct/total
        icarl_acc = 100.*correct_icarl/total
        ncm_acc = 100.*correct_ncm/total

        return [cnn_acc, icarl_acc, ncm_acc]
