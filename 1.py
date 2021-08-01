from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im = np.asarray(Image.open('1.jpg'))

im_2 = np.asarray(Image.open('2.jpg'))

black_backbround = np.ones(im_2.shape)

def cs(array):
    im = Image.fromarray(np.uint8(array))
    im.show()
    return im
    


def invert_gamma_mapping(array):
    gamma = 2.2
    gamma = 1/gamma
    # formula g(X) = f(X)^1/gamma
    array = array**gamma
    return array


value_invert_after = invert_gamma_mapping(im)

hu = cs(value_invert_after)

cs(im)

def scale_bya_factor(array,factor):
    array = array*factor
    cs(array)

def compositing(background,Foreground,alpha):
    C = (1-alpha)*background + alpha*Foreground
    return C

    
val_c = compositing(im, im_2,1)  

''' background aplha 1'''
''' foreground alpha 0 '''
cs(val_c)    

vall = compositing(im_2, black_backbround, alpha=0.55)
cs(vall)


def get_r_g_b(im):
    r = im[0]
    g = im[1]
    b = im[2]
    return r,g,b


im_copy = im.copy()


im_copy[2:] = im_copy[2:]/2
im_copy[1:] = abs(im_copy[1:] -11)
cs(im_copy)


r,g,b = get_r_g_b(black_backbround)
plt.plot(r,c='r')
plt.plot(g,c='g')
plt.plot(b,c='b')
plt.show()

f= np.ones((3,3,3))*12

fil = np.ones((2,2))*0.1


        
        




    
    
    
    
    
    
    
    
    
    
    
    