#!/usr/bin/python
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utils import *
import sys
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import warnings


class pred:
    def __init__(self):
#GLOBAL VARIABLE INITIALIZATIONS TO BUILD MODEL
        self.n_steps = 80
        self.hidden_dim = 500
        self.frame_dim = 4096
        self.batch_size = 1
        self.vocab_size = len(word2id)
        self.grap=None
        self.ses=None
        self.done=False
        self.video = None
        self.caption = None
        self.caption_mask = None
        self.dropout_prob = None
        self.output_logits=None
        self.loss=None
        self.saver=None
        self.model=None
        self.bias_init_vector = get_bias_vector()

    def extract_feat(self,frame):   
        os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        fc_feats = model.predict(frame, batch_size=128)
        img_feats = np.array(fc_feats)
        return img_feats
    

    def model_cnn_load(self):
        os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        model_final = Model(inputs=model.input, outputs=out)
        return model_final

    def build_model(self):
        os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        """This function creates weight matrices that transform:
                * frames to caption dimension
                * hidden state to vocabulary dimension
                * creates word embedding matrix """

        """print("Network config: \nN_Steps: {}\nHidden_dim:{}\nFrame_dim:{}\nBatch_size:{}\nVocab_size:{}\n".format(self.n_steps,
                                                                                                        self.hidden_dim,
                                                                                                        self.frame_dim,
                                                                                                        self.batch_size,
                                                                                                        self.vocab_size))
        """
        #Create placeholders for holding a batch of videos, captions and caption masks
        video = tf.placeholder(tf.float32,shape=[self.batch_size,self.n_steps,self.frame_dim],name='Input_Video')
        caption = tf.placeholder(tf.int32,shape=[self.batch_size,self.n_steps],name='GT_Caption')
        caption_mask = tf.placeholder(tf.float32,shape=[self.batch_size,self.n_steps],name='Caption_Mask')
        dropout_prob = tf.placeholder(tf.float32,name='Dropout_Keep_Probability')
        #For im2 cap
        with tf.variable_scope('Im2Cap') as scope:
            W_im2cap = tf.get_variable(name='W_im2cap',shape=[self.frame_dim,
                                                        self.hidden_dim],
                                                        initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
            b_im2cap = tf.get_variable(name='b_im2cap',shape=[self.hidden_dim],
                                                        initializer=tf.constant_initializer(0.0))
        with tf.variable_scope('Hid2Vocab') as scope:
            W_H2vocab = tf.get_variable(name='W_H2vocab',shape=[self.hidden_dim,self.vocab_size],
                                                             initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
            b_H2vocab = tf.Variable(name='b_H2vocab',initial_value=self.bias_init_vector.astype(np.float32))

        with tf.variable_scope('Word_Vectors') as scope:
            word_emb = tf.get_variable(name='Word_embedding',shape=[self.vocab_size,self.hidden_dim],
                                                                    initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        #print("Created weights")

        #Build two LSTMs, one for processing the video and another for generating the caption
        with tf.variable_scope('LSTM_Video',reuse=None) as scope:
            lstm_vid = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            lstm_vid = tf.nn.rnn_cell.DropoutWrapper(lstm_vid,output_keep_prob=dropout_prob)
        with tf.variable_scope('LSTM_Caption',reuse=None) as scope:
            lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            lstm_cap = tf.nn.rnn_cell.DropoutWrapper(lstm_cap,output_keep_prob=dropout_prob)

        #Prepare input for lstm_video
        video_rshp = tf.reshape(video,[-1,self.frame_dim])
        video_rshp = tf.nn.dropout(video_rshp,keep_prob=dropout_prob)
        video_emb = tf.nn.xw_plus_b(video_rshp,W_im2cap,b_im2cap)
        video_emb = tf.reshape(video_emb,[self.batch_size,self.n_steps,self.hidden_dim])
        padding = tf.zeros([self.batch_size,self.n_steps-1,self.hidden_dim])
        video_input = tf.concat([video_emb,padding],1)
        #print("Video_input: {}".format(video_input.get_shape()))
        #Run lstm_vid for 2*n_steps-1 timesteps
        with tf.variable_scope('LSTM_Video') as scope:
            out_vid,state_vid = tf.nn.dynamic_rnn(lstm_vid,video_input,dtype=tf.float32)
        #print("Video_output: {}".format(out_vid.get_shape()))

        #Prepare input for lstm_cap
        padding = tf.zeros([self.batch_size,self.n_steps,self.hidden_dim])
        caption_vectors = tf.nn.embedding_lookup(word_emb,caption[:,0:self.n_steps-1])
        caption_vectors = tf.nn.dropout(caption_vectors,keep_prob=dropout_prob)
        caption_2n = tf.concat([padding,caption_vectors],1)
        caption_input = tf.concat([caption_2n,out_vid],2)
        #print("Caption_input: {}".format(caption_input.get_shape()))
        #Run lstm_cap for 2*n_steps-1 timesteps
        with tf.variable_scope('LSTM_Caption') as scope:
            out_cap,state_cap = tf.nn.dynamic_rnn(lstm_cap,caption_input,dtype=tf.float32)
        #print("Caption_output: {}".format(out_cap.get_shape()))

        #Compute masked loss
        output_captions = out_cap[:,self.n_steps:,:]
        #print("shape of oo "+str(output_captions.shape))
        output_logits = tf.reshape(output_captions,[-1,self.hidden_dim])
        output_logits = tf.nn.dropout(output_logits,keep_prob=dropout_prob)
        output_logits = tf.nn.xw_plus_b(output_logits,W_H2vocab,b_H2vocab)
        output_labels = tf.reshape(caption[:,1:],[-1])
        caption_mask_out = tf.reshape(caption_mask[:,1:],[-1])
        #print("shape of caption_mask_out "+str(caption_mask_out.shape))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_logits,labels=output_labels)
        masked_loss = loss*caption_mask_out
        loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(caption_mask_out)
        return video,caption,caption_mask,output_logits,loss,dropout_prob

    def predict(self,video_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.done is False:
                self.grap=tf.Graph()
                with self.grap.as_default():
                    learning_rate = 0.00001
                    self.video,self.caption,self.caption_mask,self.output_logits,self.loss,self.dropout_prob = self.build_model()
                    #print(self.output_logits)
                    optim = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)
                    self.saver = tf.train.Saver()
                    self.model=self.model_cnn_load()

                self.done=True
                
            with self.grap.as_default():
                    
                    with tf.Session() as sess:
                        self.model=self.model_cnn_load()
                        ckpt_file = 'S2VT_Dyn_10_0.0001_300_46000.ckpt.meta'
                        if ckpt_file:                        
                            self.saver.restore(sess,'./S2VT_Dyn_10_0.0001_300_46000.ckpt')
                            print("Restored model")
                        else:
                            sess.run(tf.initialize_all_variables())
                        self.model=self.model_cnn_load()
                        get_output=""
                        count=0
                        times=0
                        frames= np.zeros((80, 224, 224, 3))
                        cap = cv2.VideoCapture(video_path)
                        #fps = cap.get(cv2.CAP_PROP_FPS)
                        #cap.set(cv2.CAP_PROP_FPS, 20)
                        #fps = cap.get(cv2.CAP_PROP_FPS)
                        #print("fps is "+str(fps))
                        #fra = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        #print("frames is "+str(fra))
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if ret is False:
                                break
                            img = cv2.resize(frame, (224, 224))
                            frames[count]=img
                            count+=1
                            
                            if count==80:
                                get_o=""
                                #print(frames)
                                fc_feats = self.model.predict(frames, batch_size=128)
                                #print(fc_feats)
                                img_feats = np.array(fc_feats)
                                vid = np.zeros((1,80, 4096))
                                vid[0]=img_feats
                                caps,caps_mask = convert_caption(['<BOS>'],word2id,80)
                        
                                for i in range(self.n_steps-1):
                                    #print(self.output_logits)
                                    o_l = sess.run(self.output_logits,feed_dict={self.video:vid,
                                                                    self.caption:caps,
                                                                    self.caption_mask:caps_mask,
                                                                    self.dropout_prob:1.0})
                                    #print("ol shape is  "+str(o_l.shape))
                                    out_logits = o_l.reshape([self.batch_size,self.n_steps-1,self.vocab_size])
                                    output_captions = np.argmax(out_logits,2)
                                    caps[0][i+1] = output_captions[0][i]
                                    print_in_english(caps)
                                    get_o=get_in_english(caps)
                                    if ((id2word[output_captions[0][i]] == '<EOS>')):
                                        break
                                count=0;
                                get_output=get_output+"\n"+get_in_english(caps)
                        
                        return get_output
                    
if __name__=="__main__":  
                p=pred()
                print(p.predict('C:/major project/vvv/video-captioning-master/Data/YouTubeClips/fr9H1WLcF1A_141_148.avi'))
                print(p.predict('C:/major project/vvv/video-captioning-master/Data/YouTubeClips/-_hbPLsZvvo_172_179.avi'))