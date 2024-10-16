import sys
import os 
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import smplotlib
import matplotlib.pyplot as plt
from configs.config import get_cfg_defaults
import numpy as np
from wordcloud import WordCloud
import matplotlib.gridspec as gridspec
from pathlib import Path

#get the configs
cfg=get_cfg_defaults()
subj=2
finetune=False
percent=100
backbone='alexnet'
objective_name='default'
cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'/'+backbone + '.yaml')
opts=["BACKBONE.FINETUNE",finetune,"BACKBONE.PERCENT_OF_CHANNELS",percent]
cfg.merge_from_list(opts)
cfg.freeze()
#print(cfg)


#load the dreams
roi_names=['V1','V2','floc-places','floc-faces','floc-bodies']
# roi_name=roi_names[-3]
# dream_path=str(os.path.join(cfg.PATHS.NSD_DREAMS,'subj%02d'%subj,cfg.BACKBONE.NAME,
#                             'finetune-'+str(cfg.BACKBONE.FINETUNE),
#                             'percent_channels-%d'%(int(cfg.BACKBONE.PERCENT_OF_CHANNELS)),
#                             'objective-%s'%objective_name))
# print(dream_path)                            
# out_name=dream_path +'/roi-%s'%roi_name + '_obj-%s'%objective_name + '.npy'
# imgs=np.load(out_name)


# # #plot
# # fig,ax=plt.subplots(len(imgs))
# # for i in range(0,len(imgs)):
# #     ax.flatten()[i].imshow(np.flipud(imgs[i]))
# # plt.savefig(cfg.PATHS.NSD_PLOTS + '/test.png')

# #load the words corresponding to the dream
# N=1000
# text_path= os.path.join(cfg.PATHS.NSD_TEXT, 
#                         'dreams',
#                         'subj%02d'%subj,
#                         backbone,
#                         'finetune-%s'%str(finetune),
#                         'percent_channels-%d'%percent,
#                         'objective-%s'%objective_name,
#                         'roi-%s_obj-%s_N-%d.pkl'%(roi_name,objective_name,N))
# text_dic=np.load(text_path,allow_pickle=True)

# dream_id=0
# word_dic=dict(zip(list(text_dic['nouns']),list(100*text_dic['nouns_similarity'])[dream_id].numpy()))

# sorted_word_frequencies = dict(sorted(word_dic.items(), key=lambda x: x[1], reverse=True)[:50])

# wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(word_dic)

# plt.figure(figsize=(5, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Turn off the axis
# plt.show()
# plt.savefig(cfg.PATHS.NSD_PLOTS + '/test.png')


# fig, ax = plt.subplots(1,2)
# ax[0].imshow((1.50*imgs[dream_id]))
# ax[0].axis('off')  # Turn off the axis
# ax[1].imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Turn off the axis
# plt.savefig(cfg.PATHS.NSD_PLOTS + '/test.png')



class model_params:
    def __init__(self,name,percents,finetune):
        self.name=name
        self.percents=percents
        self.finetune=finetune
        self.input_size=self.get_input_size()

    def load_dreams(self,subj,roi_name,objective_name='default'):
        dreams=[]
        for percent in self.percents:
            dream_path=str(os.path.join(cfg.PATHS.NSD_DREAMS,'subj%02d'%subj,self.name,
                            'finetune-'+str(self.finetune),
                            'percent_channels-%d'%(int(percent)),
                            'objective-%s'%objective_name))
            out_name=dream_path +'/roi-%s'%roi_name + '_obj-%s'%objective_name + '.npy'
            dream=np.load(out_name)
            dreams.append(self.dreams_processing(dream))
        return dreams

    def load_words(self,subj,roi_name,objective_name='default',N=1000,max_dream_id=4,H=600,W=600,top=25):
        words_dics_percents=[]
        for percent in self.percents:
            text_path= os.path.join(cfg.PATHS.NSD_TEXT, 
                            'dreams',
                            'subj%02d'%subj,
                            self.name,
                            'finetune-%s'%str(self.finetune),
                            'percent_channels-%d'%percent,
                            'objective-%s'%objective_name,
                            'roi-%s_obj-%s_N-%d.pkl'%(roi_name,objective_name,N))
            text_dic=np.load(text_path,allow_pickle=True)
            word_dics=[]
            for dream_id in range(0,max_dream_id):
                word_dic=dict(zip(list(text_dic['nouns']),list(100*text_dic['nouns_similarity'])[dream_id].numpy()))
                sorted_word_frequencies = dict(sorted(word_dic.items(), key=lambda x: x[1], reverse=True)[:top])
                wordcloud = WordCloud(width=W, height=H, background_color='white').generate_from_frequencies(sorted_word_frequencies)
                word_dics.append(wordcloud)
            words_dics_percents.append(word_dics)
        return words_dics_percents

    def get_input_size(self):
        cfg=self.get_config()
        return cfg.BACKBONE.INPUT_SIZE

    def get_config(self):
        cfg=get_cfg_defaults()
        cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+self.name +'.yaml')
        opts=["BACKBONE.FINETUNE",self.finetune,"BACKBONE.PERCENT_OF_CHANNELS",self.percents[0]]
        cfg.merge_from_list(opts)
        return cfg
    
    def dreams_processing(self,dream):
        #mask=mask.abs().mean(0)
        #mask=mask/mask.max()
        #mask[mask<0.1]=0
        #mask=GaussianBlur(21,sigma=(1.2,1.2))(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        #mask=np.clip(1.0*mask/mask.max(),0,1)
        #mask[mask<0.275]=0
        return dream#mask.moveaxis(0,-1)


dream_id_dic={'1':{'V1':0,'V2':0,'V3':0, 'V3ab':0,'floc-faces':1,'floc-places':1,'floc-bodies':3,'floc-words':3},
                '2':{'V1':0,'V2':0,'V3':0, 'V3ab':0,'floc-faces':0,'floc-places':0,'floc-bodies':0,'floc-words':0},
                '5':{'V1':0,'V2':0,'V3':0, 'V3ab':0,'floc-faces':0,'floc-places':0,'floc-bodies':0,'floc-words':0},
                '7':{'V1':0,'V2':0,'V3':0, 'V3ab':0,'floc-faces':0,'floc-places':0,'floc-bodies':0,'floc-words':0},
                }


#so again may be we can do subplot and grid spec
subj=1

models=[model_params('alexnet',[25,100],False),
        model_params('alexnet',[10,100],True),
        model_params('vgg11',[5,100],False),
        model_params('vgg11',[1,100],True),
        model_params('clip',[1,100],False)]
roi_names1=['floc-faces','floc-places','floc-words','floc-bodies']
roi_names2=['V1','V2','V3','V3ab']
plot_names=['late','early']
for plot_name,roi_names in zip(plot_names,[roi_names1, roi_names2]):
    cols = 5
    rows=2*len(roi_names)
    fig, ax_ = plt.subplots(rows, cols, figsize=(2*cols, rows),gridspec_kw={'wspace': 0.05, 'hspace': 0.015},dpi=1200)
    for r,roi_name in enumerate(roi_names):
        i=0
        roi_name_=roi_name.split('-')
        if len(roi_name_)>1:
            roi_name_=roi_name_[1]
        else:
            roi_name_=roi_name
        fig.text(-0.1, 0, '%s'%roi_name_, fontsize=12, ha='center', va='center',transform=ax_[2*r,0].transAxes,rotation=90)
        for m,model in enumerate(models):
            ax_[2*r, m].axis('off')
            ax_[2*r+1, m].axis('off')

            grid1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_[2*r, m].get_subplotspec(), wspace=0.02, hspace=0)
            grid2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_[2*r+1, m].get_subplotspec(), wspace=0.02, hspace=0)


            dreams=model.load_dreams(subj,roi_name)
            words=model.load_words(subj,roi_name)

            for k in range(0,len(dreams)):
                dream=dreams[k]
                word=words[k]
                ax1 = fig.add_subplot(grid1[k])
                ax2 = fig.add_subplot(grid2[k])

                ax1.axis('off')
                ax2.axis('off')

                dream_id=dream_id_dic[str(subj)][roi_name]
                ax1.imshow(np.flipud(1.25*dream[dream_id]))
                ax2.imshow(np.flipud(word[dream_id]))

                if r==0:
                    fig.text(0.25, 1.3, '%s'%((model.name).capitalize()), fontsize=12, ha='center', va='center',transform=ax_[0,i].transAxes)
                    fig.text(0.75, 1.3, '%s'%((model.name).capitalize()), fontsize=12, ha='center', va='center',transform=ax_[0,i].transAxes)
                    fig.text(0.25, 1.1, '%d-%s'%(model.percents[0],str(model.finetune)), fontsize=12, ha='center', va='center',transform=ax_[0,i].transAxes)
                    fig.text(0.75, 1.1, '%d-%s'%(model.percents[1],str(model.finetune)), fontsize=12, ha='center', va='center',transform=ax_[0,i].transAxes)
            i+=1
    file_path=Path(os.path.join(cfg.PATHS.NSD_PLOTS,'dreams','wordcloud','subj%02d'%subj))#,plot_name+'.png'))
    file_path.mkdir(parents=True,exist_ok=True)
    plt.savefig(str(file_path)+'/'+plot_name + '.png' + '',dpi=800)