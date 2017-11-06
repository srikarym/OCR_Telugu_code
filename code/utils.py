from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import numpy as np

def get_max_freq_elem( arr):

    max_arr = []
    freqs = {}
    for i in arr:
        if i in freqs:
            freqs[i] += 1
        else:
            freqs[i] = 1

    sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
    max_freq = freqs[sorted_keys[0]]

    for k in sorted_keys:
        if freqs[k] == max_freq:
            max_arr.append(k)

    return max_arr

def compare_sum(value):
    if value >= 44 and value <= 46:
        return True
    else:
        return False

# determine the skew of the image  
def determine_skew(img):
    edges = canny(img, sigma=3.0)
    h, a, d = hough_line(edges)
    _, ap, _ = hough_line_peaks(h, a, d, num_peaks=10)

    
    absolute_deviations = [np.abs(np.pi/4 - np.abs(k) ) for k in ap]
    average_deviation = np.mean(np.rad2deg(absolute_deviations))
    ap_deg = [np.rad2deg(x) for x in ap]

    bin_0_45 = []
    bin_45_90 = []
    bin_0_45n = []
    bin_45_90n = []

    for ang in ap_deg:

        deviation_sum = int(90 - ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_45_90.append(ang)
            continue

        deviation_sum = int(ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_0_45.append(ang)
            continue

        deviation_sum = int(-ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_0_45n.append(ang)
            continue

        deviation_sum = int(90 + ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_45_90n.append(ang)

    angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
    lmax = 0

    for j in range(len(angles)):
        l = len(angles[j])
        if l > lmax:
            lmax = l
            maxi = j

    if lmax:
        ans_arr = get_max_freq_elem(angles[maxi])
        ans_res = np.mean(ans_arr)

    else:
        ans_arr = get_max_freq_elem(ap_deg)
        ans_res = np.mean(ans_arr)

    data = {
        "Average Deviation from pi/4": average_deviation,
        "Estimated Angle": ans_res,
        "Angle bins": angles}
 
    return data

#determine the rotation angle
def deskew(img):
    
    res = determine_skew(img)
    angle = res['Estimated Angle']
    if angle >= 0 and angle <= 45:
        rot_angle = 90 - angle
    if angle >= 45 and angle <= 90:
        rot_angle = angle - 90
    if angle >= -45 and angle < 0:
        rot_angle = angle + 90
    if angle >= -90 and angle < -45:
        rot_angle = 90 + angle
    return rot_angle

# cropping the image
def crop(img):

    height,width = img.shape

    # Histogram along x
    vec = []
    for k in xrange(height):
        count = 0
        for j in xrange(width):
            if img[k,j]!=255:
                count = count + 1
        vec.append(count)
    # Left zeros
    count_left = 0
    for k in xrange(len(vec)):
        if vec[k]==0:
            count_left = count_left + 1
        else :
            break
    # Right Zeros
    count_right = 0
    for k in xrange(len(vec)):
        if vec[len(vec)-k-1]==0:
            count_right = count_right + 1
        else:
            break
    img = img[count_left:height-count_right,:]
    height,width = img.shape
    vec = []
    for k in xrange(width):
        count = 0
        for j in xrange(height):
            if img[j,k]!=255:
                count = count + 1
        vec.append(count)
    # Left zeros
    count_left = 0
    for k in xrange(len(vec)):
        if vec[k]==0:
            count_left = count_left + 1
        else :
            break
    # Right Zeros
    count_right = 0
    for k in xrange(len(vec)):
        if vec[len(vec)-k-1]==0:
            count_right = count_right + 1
        else:
            break
    img = img[:,count_left:width-count_right]
    return img
