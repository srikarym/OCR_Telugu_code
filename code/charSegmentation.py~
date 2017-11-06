import cv2
import numpy as np
from copy import deepcopy

# resolve the equivalence in the components resulted from connected components function
def resolve_equivalence(im, width, height, counter):
    equi = np.array(np.zeros(counter+1))
    for i in range(len(equi)):
        equi[i] = i
    for i in range(height):
        for j in range(width):
            if j == 0:
                l = 0
            else:
                l = 1
            if i == 0:
                n = 0
            else:
                n = 1
            x = im[i-n][j]
            y = im[i][j-l]
            if (x != y) and (x != 0) and (y != 0):
                while (equi[x] != x):
                    x = int(equi[x])
                while (equi[y] != y):
                    y = int(equi[y])
                equi[x] = min(equi[x],equi[y])
                equi[y] = equi[x]
    for i in range(len(equi)):
        x = i
        while(equi[x] != x):
            x = int(equi[x])
        equi[i] = x
    for i in range(height):
        for j in range(width):
            im[i][j] = equi[im[i][j]]
    return im

# group the components in order wise 1,2..., n
def grouping(im, width, height, counter):
    gp = [0] * (counter + 1)
    for i in range(height):
        for j in range(width):
            gp[im[i][j]] += 1
    val = 0
    for i in range(len(gp)):
        if gp[i] >= 1:
            gp[i] = val
            val += 1
    for i in range(height):
        for j in range(width):
            im[i][j] = gp[im[i][j]]
    return im, val

# connected components algorithm 
def connected_components(im, width, height):
    con_img = np.array([[0 for i in range(width)] for j in range(height)])
    counter = 1
    for i in range(height):
        for j in range(width):
            if j == 0:
                l = 0
            else:
                l = 1
            if i == 0:
                n = 0
            else:
                n = 1
            if (im[i][j] == 0) and (im[i][j-l] == 255) and (im[i-n][j] == 255):
                con_img[i][j] = counter
                counter += 1
            elif (im[i][j] == 0) and (im[i][j-l] == 255) and (im[i-n][j] == 0):
                con_img[i][j] = con_img[i-n][j]
            elif (im[i][j] == 0) and (im[i][j-l] == 0) and (im[i-n][j] == 255):
                con_img[i][j] = con_img[i][j-l]
            elif (im[i][j] == 0) and (im[i][j-l] == 0) and (im[i-n][j] == 0):
                con_img[i][j] = con_img[i-n][j]
    final_img = resolve_equivalence(con_img, width, height, counter)   
    img, components = grouping(final_img, width, height, counter)
    return img, components

def complete(Image):

    # Textportion and positions correspondingly in the original image
    Text_regions = []
    pos = []
    
    # Thresholding
    ret,Thresh_1 = cv2.threshold(Image,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    Thresh_2 = np.array([[255 for i in range(Thresh_1.shape[1]+2)] for j in range(Thresh_1.shape[0]+2)])

    for i in range(Thresh_1.shape[0]):
        for j in range(Thresh_1.shape[1]):
            Thresh_2[i+1][j+1] = Thresh_1[i][j]
            
    height,width = Thresh_2.shape
    
    # Getting the connected components        
    img, components = connected_components(Thresh_2, width, height)
    
    minw = [width+5] * (components)
    minh = [height+5] * (components)
    maxw = [0] * (components)
    maxh = [0] * (components)	
    
    
    # Finding maximum and minimum heights and widths of the components
    weight = [0] * (components)
    count = 0
    for i in range(height):
        for j in range(width):
            if img[i][j] != 0:
                count += 1
                minw[img[i][j]] = min(minw[img[i][j]], j)
                maxw[img[i][j]] = max(maxw[img[i][j]], j)
                weight[img[i][j]] += 1
    for i in range(width):
        for j in range(height):
            if img[j][i] != 0:
                minh[img[j][i]] = min(minh[img[j][i]], j)
                maxh[img[j][i]] = max(maxh[img[j][i]], j)
                
    avg = (0.3*(count/components))
     
    # Grouping vattus and matras with the corresponding character            
    groups = [0] * (components)
    prob = [0] * (components)
    gp_num = 1
    for i in range(len(groups)):
        if i != 0:
            if (groups[i] == 0) and (maxw[i] > minw[i]) and (maxh[i] > minh[i]) and (weight[i] > avg):
                groups[i] = gp_num
                for j in range(len(groups)):
                    if (maxw[j] > minw[j]) and (maxh[j] > minh[j]) and (weight[j] > avg): 
                        if (minw[i] >= minw[j] and maxw[i] <= maxw[j]) or (minw[j] >= minw[i] and maxw[j] <= maxw[i]):
                            groups[j] = gp_num
                        elif (minw[j] >= minw[i] and minw[j] <=maxw[i]) or (maxw[j] >= minw[i] and maxw[j] <= maxw[i]):
                            common_dist = float(min(maxw[j],maxw[i]) - max(minw[j],minw[i]))
                            min_dist = float(min(maxw[j]-minw[j],maxw[i]-minw[i]))
                            if common_dist >= (0.3*min_dist):
                                if (groups[j] != 0) and (common_dist > prob[j]):
                                    groups[j] = gp_num
                                    prob[j] = common_dist
                                else:
                                    groups[j] = gp_num
                                    prob[j] = common_dist
                gp_num += 1
                
    # Getting combined width and height with character and corresponding matras 
    com_minw = [width+5] * (gp_num)
    com_maxw = [0] * (gp_num)
    com_minh = [height+5] * (gp_num)
    com_maxh = [0] * (gp_num)
    
    for i in range(len(groups)):
        if groups[i] != 0:
            com_minw[groups[i]] = min(com_minw[groups[i]],minw[i])
            com_minh[groups[i]] = min(com_minh[groups[i]],minh[i])
            com_maxw[groups[i]] = max(com_maxw[groups[i]],maxw[i])
            com_maxh[groups[i]] = max(com_maxh[groups[i]],maxh[i])
    
    # Marking final positions        
    for i in range(len(com_minw)):
        if i!=0:
            pos.append([com_minh[i]-1,com_minw[i]-1,com_maxh[i]+1,com_maxw[i]+1])

    pos = sorted(pos, key=lambda y: y[1])
    
    # Marking final Text regions and removing minor blobs     
    for j in range(len(pos)):
        temp = [0] * (components)
        Text_regions.append(deepcopy(Thresh_2[pos[j][0]:pos[j][2],pos[j][1]:pos[j][3]])) 
        for i in range(pos[j][2]-pos[j][0]):
            for k in range(pos[j][3]-pos[j][1]):
                temp[groups[img[i+pos[j][0]][k+pos[j][1]]]] += 1
        max_ = 0
        for i in range(len(temp)):
            if i != 0:
                max_ = max(max_,temp[i])
        for i in range(pos[j][2]-pos[j][0]):
            for k in range(pos[j][3]-pos[j][1]):
                if temp[groups[img[i+pos[j][0]][k+pos[j][1]]]] != max_:
                    Text_regions[j][i][k] = 255
                    
    return (pos,Text_regions)
############################################################################################################################## 
