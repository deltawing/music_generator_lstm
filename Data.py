# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:50:27 2018

@author: 罗骏
"""

import tensorflow as tf
import pretty_midi as pm
import numpy as np
import pathlib
import random
import os
from tqdm import tqdm

#CHANNEL_NUM = 2
CLASS_NUM = 72
INPUT_LENGTH = 500

def roll(path):
    try:
        song = pm.PrettyMIDI(midi_file=str(path), resolution=96)
    except:
        tqdm.write('Error while opening')
        raise Exception
    #index = [0, 1]
    piano_rolls = [i.get_piano_roll()[24:96] for i in song.instruments]
    length = np.min([i.shape[1] for i in piano_rolls])
    if length < INPUT_LENGTH:
        tqdm.write('Too short')
        raise Exception
    data = np.zeros((CLASS_NUM, length))
    for piano_roll, instrument in zip(piano_rolls, song.instruments):
        if not instrument.is_drum:
            data = np.add(data, piano_roll[:, :length])
        else:
            continue            
    if np.max(data) == 0:
        tqdm.write('No notes')
        raise Exception
    data = data > 0
    data = np.split(data[:, :length // INPUT_LENGTH * INPUT_LENGTH], 
                    indices_or_sections=length // INPUT_LENGTH, axis=-1)  
    return data

def build_dataset():
    pathlist = list(pathlib.Path('Classics').glob('**/*.mid'))
    random.shuffle(pathlist)
    if not os.path.exists('Dataset'):
        os.mkdir('Dataset')
    pathlist_1 = pathlist[:len(pathlist)//4]
    pathlist_2 = pathlist[len(pathlist)//4:len(pathlist)//2]
    pathlist_3 = pathlist[len(pathlist)//2:3*len(pathlist)//4]
    pathlist_4 = pathlist[3*len(pathlist)//4:]
    data_row = 0
    for i, count in zip([pathlist_1, pathlist_2, pathlist_3, pathlist_4],[1,2,3,4]):
        writer = tf.python_io.TFRecordWriter('Dataset/dataset_'+str(count)+'.tfrecord')
        for path in tqdm(i):
            try:
                data = roll(str(path))
            except:
                continue
            for datum in data:
                data_row += 1
                datum = np.packbits(datum).tostring()
                feature = {'roll': tf.train.Feature(bytes_list=tf.train.BytesList(value=[datum]))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized = example.SerializeToString()
                writer.write(serialized)
        writer.close()
        print('Dataset/dataset_'+str(count)+'.tfrecord', data_row, 'rows')
if __name__ == '__main__':
    build_dataset()