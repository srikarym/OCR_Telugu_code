# importing libraries
import cv2
import numpy as np
import imghdr
import optparse
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
import os
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import collections
from scipy.signal import argrelextrema
from PIL import Image
# Keras with tensorflow backend
from keras.models import load_model
from keras.models import model_from_json
import json
from numpy.lib.stride_tricks import as_strided
import json
import socket
import struct
import charSegmentation
import utils

############################################################################################################################## 
# Model and some files loading part
############################################################################################################################## 

def file_char_vattu_gunintam(file_name):
    file_1 = open(file_name,'r')
    part_1_1 = []
    part_1_2 = []

    part_1_1.append(file_1.readline())
    k = file_1.readline()
    while k != '' :
        part_1_2.append(k)
        part_1_1.append(file_1.readline())
        k = file_1.readline()

    if part_1_2[len(part_1_2)-1]=='':
        del part_1_2[-1]

    if part_1_1[len(part_1_1)-1]=='':
        del part_1_1[-1]

    for i in xrange(len(part_1_1)):
        part_1_1[i] = int(part_1_1[i])
    return (part_1_1,part_1_2)

# Model for only characters
with open('/models/main_character/ours/model_chars.json') as infile:
    json_char = json.load(infile)

model_1 = model_from_json(json_char)
model_1.load_weights('/models/main_character/ours/model_chars_weights.hdf5')

# Model for vattulu and gunintalu
with open('/models/vattu_gunintam/ours/model_v_g.json') as infile:
    json_char = json.load(infile)

model_2 = model_from_json(json_char)
model_2.load_weights('/models/vattu_gunintam/ours/model_v_g_weights.hdf5')

(char_1,char_2) = file_char_vattu_gunintam('char.txt')

(vattu_1,vattu_2) = file_char_vattu_gunintam('vattu_gunintam.txt')

print 'Loaded Model File!!!'

############################################################################################################################## 
# Server part to send and recirve data from app
############################################################################################################################## 

address1 = ("192.168.43.25", 8891)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address1)
s.listen(1000)

address2 = ("192.168.43.25", 8801)
s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s2.bind(address2)
s2.listen(1000)

############################################################################################################################## 
# Continuously listen for image data from app
##############################################################################################################################

while(1):
    client, addr = s.accept()
    print "got connected from", addr
    buf = ''
    while len(buf)<4:
        buf += client.recv(4-len(buf))
            
    size = struct.unpack('!i', buf)
    print "receiving %s bytes" % size
    with open('img.jpg', 'wb') as img:
        while True:
            data = client.recv(1024)
            if not data:
                break
            img.write(data)

    print "Received image from app"
    
############################################################################################################################## 
# Image Processing and recognition part
############################################################################################################################## 

    file_name = 'img.jpg'
    img = np.asarray(cv2.imread(file_name,0))
    kernel = np.ones((9,9),np.uint8)
    erode = cv2.erode(img,kernel,iterations = 1)
    angle = utils.deskew(erode)
    rows,cols = img.shape
    img = cv2.imread(file_name,0)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),int(angle),1)
    gray_scale = cv2.warpAffine(img,M,(cols,rows))

    # mser properties
    _delta=5
    _min_area=60
    _max_area=14400
    _max_variation=0.25
    _min_diversity=.2
    _max_evolution=200
    _area_threshold=1.01
    _min_margin=0.003
    _edge_blur_size=5

    mser = cv2.MSER_create(_delta,_min_area,_max_area,_max_variation,_min_diversity,_max_evolution,_area_threshold,_min_margin,_edge_blur_size)

    regions = mser.detectRegions(gray_scale,None)

    out_image_2 = np.zeros(gray_scale.shape,dtype='uint8')

    bool_idx = []
    regions_2 = []
    areas_regions = []
    for i in xrange(len(regions)):
        kk = np.asarray(regions[i])
        min_1 = np.amin(kk[:,0])
        max_1 = np.amax(kk[:,0])
        min_2 = np.amin(kk[:,1])
        max_2 = np.amax(kk[:,1])
        ratio = float(len(regions[i]))/((max_2-min_2)*(max_1-min_1))       
        if max_2==min_2 or max_1==min_1:
            bool_idx.append(False)
        else:
            if (not(float(max_2-min_2)/float(max_1-min_1) < 0.1 or float(max_1-min_1)/float(max_2-min_2) <0.1 or ratio<0.2)):
                out_image_2[ kk[:,1] , kk[:,0] ] = 255
                areas_regions.append((max_2-min_2)*(max_1-min_1))
                regions_2.append(regions[i])
                bool_idx.append(True)
            else :
                bool_idx.append(False)

    areas_regions = np.asarray(areas_regions)

    regions = regions_2

    n,bins,patches=plt.hist(areas_regions,bins="auto")

    average = 0
    num = 0
    for i in xrange(len(areas_regions)):
        if areas_regions[i]>bins[np.argmax(n)] and areas_regions[i]<bins[np.argmax(n)+1]:
            average = average + areas_regions[i]
            num = num + 1
    average = average/float(num)


    kernell = np.ones((1,int(0.7*np.sqrt(average))),np.uint8)
    appx_size = int(0.7*np.sqrt(average))
    out_image_3 = cv2.dilate(out_image_2,kernell,iterations=1)
    kernell = np.ones((int(0.2*np.sqrt(average)),1),np.uint8)
    out_image_3 = cv2.dilate(out_image_3,kernell,iterations=1)

    out_image_4 = out_image_3[:,:]
    (_,cnts, _) = cv2.findContours(out_image_4.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    out_image_6 = deepcopy(np.asarray(gray_scale))
    regions1 = []

    for i in xrange(len(cnts)):
        x,y,w,h = cv2.boundingRect(cnts[i])
        
        include = True
        
        for j in xrange(len(cnts)):
            if j!= i:
                x1,y1,w1,h1 = cv2.boundingRect(cnts[j])
                if x>=x1 and y>=y1 and x+w<=x1+w1 and y+h<=y1+h1:
                    include = False

        if (h>2*appx_size or w>2*appx_size or w*h>100) and include:
            cv2.rectangle(out_image_6,(x,y),(x+w,y+h),(255),3)
            regions1.append([x,y,w,h])
            
    cv2.imwrite('output/region_seg.png',out_image_6)
    regions1 = np.array(regions1)
    regions1 = regions1[np.argsort(regions1[:, 1])]

    regions2 = [[] for i in range(len(regions1))]
    regions2[0].append(regions1[0])
    line_idx = 0

    for i in xrange(1,len(regions1)):
        x,y,w,h = regions1[i]
        xa,ya,wa,ha = regions1[i-1]
        a = max(y,ya)
        b = min(h+y,ha+ya)
        if(b-a)>0:
            regions2[line_idx].append(regions1[i])
        else:
            line_idx = line_idx + 1
            regions2[line_idx].append(regions1[i]) 
    regions2 = np.array(regions2)
    regions2 = [x for x in regions2 if x != []]

    for i in xrange(len(regions2)):
        newline = np.array(regions2[i])
        newline = newline[np.argsort(newline[:, 0])]
        regions2[i] = newline
    new_regions = []    
    for i in xrange( len(regions2)):
        for j in xrange(len(regions2[i])):
            new_regions.append(regions2[i][j])

    positions = []
    Text_regions = []
    k = []

    line_idx = np.zeros((len(new_regions),len(new_regions)))
    p = np.asarray(new_regions)

    new_regions_3 = np.zeros(p.shape)
    aa = np.argsort(p[:, 1])
    for i in xrange(len(new_regions_3)):
        new_regions_3[i] = new_regions[aa[i]]

    for i in xrange(len(new_regions_3)):
        for j in xrange(len(new_regions_3)):
            max_1 = max( new_regions_3[i][1] , new_regions_3[j][1] )
            min_1 = min( new_regions_3[i][3] + new_regions_3[i][1] , new_regions_3[j][3] + new_regions_3[j][1] )
            if min_1-max_1 > ((new_regions_3[i][3]) + (new_regions_3[j][3]))/4.0:
                line_idx[i,j] = 1

    new_regions_update = []    

    indexer = np.zeros(len(new_regions_3))
    for i in xrange(len(new_regions_3)):
        count = 0
        for j in xrange(len(new_regions_3)):
            if line_idx[j,i]==1:
                indexer[i] = indexer[i] + new_regions_3[j][1]
                count = count + 1
        indexer[i] = indexer[i]/float(count)

    kko =  []
    kko.append(0)
    count = 0
    checker = np.zeros(len(new_regions_3))
    for i in xrange(len(new_regions_3)):
        for j in xrange(len(new_regions_3)):
            if checker[j]==0 and line_idx[i,j]==1:
                new_regions_update.append(new_regions_3[j])
                checker[j] = 1
                count  = count + 1
        kko.append(count)

    for i in xrange(len(kko)-1):
        if kko[i+1]-kko[i]!=1 and kko[i+1]-kko[i]!=0:
            part = np.asarray(new_regions_update[kko[i]:kko[i+1]])
            part = part[np.argsort(part[:, 0])]
            new_regions_update[kko[i]:kko[i+1]] = part
        elif kko[i+1]-kko[i]==1:
            part = np.asarray(new_regions_update[kko[i]:kko[i+1]])
            new_regions_update[kko[i]:kko[i+1]] = part

    for i in xrange(len(regions1)):
        x,y,w,h = new_regions_update[i]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        (positions1,Text_regions1) = charSegmentation.complete(deepcopy(gray_scale[y:y+h,x:x+w]))
        for j in xrange(len(positions1)):
            positions1[j][0] = np.clip(positions1[j][0],0,np.inf) + y
            positions1[j][1] = np.clip(positions1[j][1],0,np.inf) + x
            positions1[j][2] = np.clip(positions1[j][2],0,np.inf) + y
            positions1[j][3] = np.clip(positions1[j][3],0,np.inf) + x
            x1,y1,x2,y2 = positions1[j]
            positions.append(positions1[j])
            Text_regions.append(Text_regions1[j])
            if j!=len(positions1)-1:
                k.append(0)
        
        if not (len(positions1)<1):
            k.append(1)

    positions = (positions)
    Text_regions = (Text_regions)

    order = sorted(range(len(positions)),key=lambda k :positions[k])
    line_idx = np.zeros((len(positions),len(positions)))

    for i in xrange(len(positions)):
        for j in xrange(len(positions)):
            max_1 = max( positions[i][0] , positions[j][0] )
            min_1 = min( positions[i][2] , positions[j][2] )

            if min_1-max_1 > ((positions[i][2]-positions[i][0]) + (positions[j][2]-positions[j][0]))/5.0:
                line_idx[i,j] = 1


    corresponding_cluster = np.zeros(len(positions))

    for i in xrange(len(positions)-1):
        if not(line_idx[i,i+1]==1 and line_idx[i+1,i]==1):
            corresponding_cluster[i+1] = 1

    chars = []
    vattu_gunintam = []
    output = []

    for i in xrange(len(Text_regions)):
        # segmentation of character
        img = utils.crop(Text_regions[i])
        cv2.imwrite('output/segmentedChars/'+str(i)+'.PNG',img)
        img = Image.open('output/segmentedChars/'+str(i)+'.PNG')
        img.load()
        if img.size!=(32,32):
            img = img.resize((32,32),Image.ANTIALIAS)
        img = np.asarray(img).reshape(1,1,32,32)
        img = img.astype('float32')
        img = img/255.0

        # Prediction of char
        out =  model_1.predict(img)
        output = [output,np.where(out==out.max())[1][:]+1]
        chars.append(np.where(out==out.max())[1][:]+1)

        # Prediction of vattu or gunintam if necessary
        if np.where(out==out.max())[1][:]+1>=20 and np.where(out==out.max())[1][:]+1<=55:
            out2 = model_2.predict(img)
            vattu_gunintam.append(np.where(out2==out2.max())[1][:]+1)
        else:
            vattu_gunintam.append(-1)

    file = open('output/result.html','w')
    
    # writing output in html format 

    for i in xrange(len(chars)):
        if i>0:
            if corresponding_cluster[i] == 1:
                file.write('<br/>')
            elif k[i-1]==1 :
                file.write('&#32;')
        file.write((char_2[int(chars[i])-1][:-1]))
        if vattu_gunintam[i]!=-1 and vattu_gunintam[i] !=1:
            file.write(vattu_2[int(vattu_gunintam[i])-1][:-1])

    file.close()
    
    # processing of speech
    os.system("espeak -m -v te -s 100 -f output/result.html -w output/speech.wav")
    
    break

############################################################################################################################## 
#       Sending the speech file and html output file to mobile application
##############################################################################################################################

    file1 = open("output/speech.wav", "rb")
            
    line = file1.read(1000)
    q = ""
    while (l):
        q += line
        line = file1.read(1000)
    client.send(q)
    file1.close() 
    print "Sent .mp3\.wav file sucessfully"

    client.close()
    
    client2, addr2 = s2.accept()
    print "got connected from", addr2
    
    file2 = open("output/result.html", "rb")

    line = file2.read(1000)
    q = ""
    while (l):
        q+= line
        line = file2.read(1000)
                
    client2.send(q)
    file2.close()
    print "Sent HTML file sucessfully"
        
    client2.close()

