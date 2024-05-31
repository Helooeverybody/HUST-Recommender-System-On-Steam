import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import NCF, NeuMF, NCF_Features

class Predict:
    '''impliment predict score value interation between user and item from three model NCF, NeuMF, and NCF with feature. Note that 
    NCF and NeuMF training with implicit feedback and NCF features training explicit feedback 
    '''
    def __init__(self, name_model = 'NCF',title = False,label = False,user_id = None):
        print('Load ....')
        self.name = name_model 
        self.user_id = user_id 
        self.title = title 
        self.label = label 

        self.NCF_Model = NCF.getmodel()
        self.NeuMF_Model = NeuMF.getmodel()
        self.NCF_Features_Model = NCF_Features.getmodel()

        self.path_NCF = '/HUST-Recommender-System-On-Steam/CF/Latent_Factor_Model/Deep_based/model_path/model_NCF_implicit.pth'
        self.path_NeuMF = '/HUST-Recommender-System-On-Steam/CF/Latent_Factor_Model/Deep_based/model_path/model_NeuMF_implicit.pth'
        self.path_NCF_Features = '/HUST-Recommender-System-On-Steam/CF/Latent_Factor_Model/Deep_based/model_path/model_NCF_Feature.pth'

        self.data_user = pd.read_pickle('/HUST-Recommender-System-On-Steam/CF/Latent_Factor_Model/Deep_based/data_process/data_user.pkl')
        self.data_item = pd.read_pickle('/HUST-Recommender-System-On-Steam/CF/Latent_Factor_Model/Deep_based/data_process/data_item.pkl')
        self.idx_user = pd.read_pickle('index_user.pkl')
        self.idx_item = pd.read_pickle('index_item.pkl')

        self.data_rec =  pd.read_pickle('/HUST-Recommender-System-On-Steam/CF/Latent_Factor_Model/Deep_based/data_process/data_rec.pkl')
        if self.name == 'NCF_Feature':
            self.data_rec_feature = pd.read_pickle('/HUST-Recommender-System-On-Steam/CF/Latent_Factor_Model/Deep_based/data_process/total_data.pkl')

    def get_predict(self):

        user = self.idx_user[self.idx_user['user_id'] == self.user_id]['idx_user'].iloc[0]
        lst_item = list(self.data_rec[self.data_rec['user_id'] == self.user_id]['app_id'].unique())

        item = list(self.idx_item[self.idx_item['app_id'].isin(lst_item)]['idx_item'])

        print('set up model .... ')
        if self.name == 'NCF':
            model = self.NCF_Model 
            model.load_state_dict(torch.load(self.path_NCF))

        if self.name == 'NeuMF':
            model = self.NeuMF_Model
            model.load_state_dict(torch.load(self.path_NeuMF))

        if self.name == 'NCF_Feature':
            model = self.NCF_Features_Model
            model.load_state_dict(torch.load(self.path_NCF_Features)) 
        
        try:
            model.eval()
            score_item = {}
            print('predict ....')

            # implicit model
            if self.name != 'NCF_Feature':
                user = torch.Tensor([user])
                for i in item:
                    if i != 0:
                        score = model(user.long(),torch.Tensor([i]).long())
                        score_item.update({i:score.item()})
                
                score_item = dict(sorted(score_item.items(), key = lambda x: -x[1]))

                top_item = list(score_item.keys())
                score_item = list(score_item.values())

                top_id_item = [self.idx_item[self.idx_item['idx_item'] == i]['app_id'].iloc[0] for i in top_item]
            else:
            # explicit model 
                
                lst_item = self.data_rec_feature[self.data_rec_feature['idx_user'] == user]
                
                for i in range(len(lst_item)):
                    u,it,f_u,f_it,t_u,t_it = lst_item.iloc[i]['idx_user'], lst_item.iloc[i]['idx_item'],lst_item.iloc[i]['features_user'],lst_item.iloc[i]['features_item'],lst_item.iloc[i]['tags_user'],lst_item.iloc[i]['tags_item']
        
                    score = model(torch.Tensor([u]).long(),torch.Tensor([it]).long(),torch.Tensor(f_u).unsqueeze(0),torch.Tensor(f_it).unsqueeze(0),torch.Tensor(t_u).unsqueeze(0),torch.Tensor(t_it).unsqueeze(0))
                    score_item.update({it:score.item()})
            
            
                score_item = dict(sorted(score_item.items(), key = lambda x: -x[1]))
            

                top_item = list(score_item.keys())
                score_item = list(score_item.values())

                top_id_item = [self.data_rec_feature[self.data_rec_feature['idx_item'] == i]['app_id'].iloc[0] for i in top_item]
            
            return [top_id_item,score_item]
        except:
            return 'You must enter exactly name of model including NCF,NeuMF,NCF_Feature'

    def get_label(self,item_id):
        label = []
        a = list(self.data_rec['app_id'].unique())

        if self.name == 'NCF_Feature':
            for i in item_id:
                if i in a :
                    label.append(self.data_rec_feature[(self.data_rec_feature['user_id'] == self.user_id) & (self.data_rec_feature['app_id'] == i)]['rating'].iloc[0])
                else:
                    label.append('?')
        else:
            for i in item_id:
                if i in a:
                    label.append(self.data_rec[(self.data_rec['user_id'] == self.user_id) & (self.data_rec['app_id'] == i)]['is_recommended'].iloc[0])
                else:
                    label.append('?')
        return label
    def get_title(self,item_id):
        title = []
        for t in item_id:
            title.append(self.data_item[self.data_item['app_id'] == t]['title'].unique())
        return title
    def get_result(self):
        result = self.get_predict()
        
        if len(result) == 2:
            top_item, score_item = result
            
        else:
            return result 

        if self.label == True:
            label = self.get_label(top_item)
        if self.title == True:
            title = self.get_title(top_item)

        if self.label and self.title:
            for i, t , l , s in zip(top_item, title, label, score_item):
                print('item_id : '+ str(i) + '| ' + 'title: ' + '| '+'label: ' + str(l) + '| ' + 'score: '+ str(s))

        elif not self.label and not self.title:
            for i, s in zip(top_item, score_item):
                print('item_id : '+ str(i) + '| ' + 'score: '+ str(s))
        elif self.label:
        
            for i, l , s in zip(top_item, label, score_item):
                print('item_id : '+ str(i) + '| ' + 'label: ' + str(l) + '| ' + 'score: '+ str(s))
        else:
            for i, t , s in zip(top_item, title, score_item):
                print('item_id : '+ str(i) + '| ' + 'title: ' + str(t) + '| ' + 'score: '+ str(s))


def predict(model_name = 'NCF',title = False,label = False,user_id = None):

    A = Predict(name_model= model_name,title= title,label = label,user_id = user_id)

    return A.get_result()

if __name__ == '__main__':
    predict(model_name= 'NCF_Feature', title = False ,label= True ,user_id= 1663073)

    #  13099571, 8531123, 1663073, 
