'''

In this module, the origin is the top left corner of the image
x : horizontal coord
y : vertical coord


A pose is encoded with a quadruplet (x, y, theta, scale) where theta in rad.
The new positions of the vertices is obtained by
 1 - apply a rotation of angle 'theta' around vertex 0.
 2 - scale the pattern by 'scale'
 3 - translate the pattern to put vertex 0 at position (x,y) 
    

     
'''
import numpy as np
import matplotlib.pyplot as plt

#import matplotlib.animation as animation

#from skimage.morphology import medial_axis
import cv2
import skimage

# default shape for the created float image
default_imf_shape = (100,200) # 100 rows, 200 columns


#------------------------------------------------------------------------------

def scan_segment(P1, P2, imf=None):
    """
    Scan the segment between the endpoints P1 and P2.
    Compute the arrays 'X', 'Y' and  'S' that consist repectively of
     -  the x-coords (column) of the pixels on the segment 'P1P2'
     -  the y-coords (row) of the pixels on the segment 'P1P2'
     -  the pixel intensities along the segment 'P1P2' 
         
       That is, S[i] is the value  of the ith pixel at coord X[i], Y[i] along 
       the segment 'P1P2'

    If 'imf' is None, the returned 'S' is also None.
    If 'imf' is not None, the returned 'X' and 'Y' are filtered to 
    the domain of 'imf'. That is,  points outside the domain are simply ignored.
        
    The x,y coords follow the convention explained in the 
    module header comment.
    
    @param
    - P1 : array-like : first point (x,y) , 
    - P2 : array-like : second point (x,y)
    - imf : image : the image being processed (single or multi-channels)
    
    @return 
       X, Y, S, all_inside
       where 
             X is 1D array of the x coord of the pixels on the segment
             Y is 1D array of the y coord of the pixels on the segment
             S is 1D array of the pixel values scanned on the segment  
             all_inside is True iff segment fully contained in 'imf'
    """
    # Make sure P1 and P2 are np arrays
    P1 = np.array(P1,dtype=float)
    P2 = np.array(P2,dtype=float)
    # n : number of the pixels that will be scan    
    n = int( np.ceil(np.linalg.norm(P2-P1)) )
    X = np.linspace(P1[0],P2[0],n).astype(int)
    Y = np.linspace(P1[1],P2[1],n).astype(int)    
    if imf is None:
        return X,Y, None, True
    # otherwise
    #    print('debug X = {}',X)
    #    print('debug Y = {}',Y)
    insiders = (0<=Y) & (Y<imf.shape[0]) & (0<=X) & (X<imf.shape[1]) 
    all_inside = insiders.all()
    X , Y = X[insiders], Y[insiders]
    if imf.ndim == 2:        
        S = imf[Y,X]    
    else:
        S = imf[Y,X,:] # multi-channel image    
        S = S.reshape((-1,imf.shape[2]))    
    return X, Y, S, all_inside
    
#-----------------------------------------------------------------------------

class Pattern(object):
    '''
    A 2D pattern specified by its vertices and edges.
    In this context, a pattern is a small graph defined by its vertices and
    edges.
    '''
    def __init__(self,V,E):
        '''
        @param
            V : array-like 2D list of coordinates of the vertices
            E : array-like list of edges (i,j) where i,j are vertex indices
        '''
        self.V = np.array(V) # x,y coords of the vertices
        self.E = np.array(E) # edge list
        self.v0_color = [1,0,0] # default color of vertex 0
        self.edge_color = [0,1,0]
        self.v_color = [0.5,0,0.5]
        
    def __str__(self):
        '''
        String representation of this pattern
        '''
        return 'V =\n {} \nE =\n {}'.format(str(self.V),str(self.E))
    
    def draw_ax(self, ax=None, pose=None, edge_only=True ):
        '''
        Draw the pattern on an matplotlib axis.
        @param
            ax : an axis to draw the pattern. 
                 Will create one and return 'ax' is None.
        '''
        if ax is None:
            ax = plt.axes()
        #    
        if pose is None:
            Vp = self.V
        else:
            Vp = self.pose_vertices(pose)
        #    
#        ax.set_ylim( max(ax.get_ylim()), min(ax.get_ylim()) ) 
        for i,j in self.E:
            #ax.plot(self.V[[i,j],0],self.V[[i,j],1], '-', color = self.edge_color)
            ax.plot(Vp[[i,j],0],Vp[[i,j],1], '-', color = self.edge_color)
        if edge_only:
            return ax
        for v in Vp:
            ax.plot(Vp[0][0],Vp[0][1], 'o',color = self.v_color)
        ax.plot(Vp[:,0],Vp[:,1], 'o',color = self.v_color)
        ax.plot(Vp[0][0],Vp[0][1], 'o',color = self.v0_color)            
        return ax

    def draw_im(self, pose, imf, color=1):
        '''
        Draw the edges of this pattern in pose 'pose' on the float image 'imf'
        @param
           pose : pose (details in module header comments)
           imf : one channel image
        '''
        Vp = self.pose_vertices(pose)

        for i,j in self.E:
            X,Y, _, _ = scan_segment(Vp[i], Vp[j], imf)
            if imf.ndim ==2:
                imf[Y,X] = 1
            else:
                imf[Y,X,:] = color

    def evaluate(self, imf, pose):
        '''
          Score this pattern at pose 'pose' with respect to the 
          cost matrix given by the 2D float array 'imf'.
          The score is mean squared distance to an edge pixel.
          The score returned is  np.inf if some of the vertices 
          corresponding to the pose are outside of the image 'imf'
          
          @return
            score : the score of 'pose' 
            Vp : the position of the vertices in 'pose'
        
        '''
        Vp = self.pose_vertices(pose)
        
        # if any outside return np.inf
        score = 0      # total number of pixels scanned inside 'imf'
        num_points = 0
        for i,j in self.E:
            X, Y, S, all_inside = scan_segment(Vp[i], Vp[j], imf)
            num_points += len(S)
            if not all_inside:
                return np.inf, Vp
            score += np.sum(S)  # L1 norm    
            #score = max(score,S.max())  # L_inf norm
            if num_points == 0:
                print("DIVIDE BY 0 ERROR YO")
        return score/num_points, Vp
        
    def footprint(self, pose):
        '''
            Return the bounding box of the pattern in the pose 'pose'
            @param 
                pose : pose (details in module header comments)
            @return
                minX , maxX, minY, maxY
        '''
        Vp = self.pose_vertices(pose)
        minX , maxX, minY, maxY = Vp[:,0].min() , Vp[:,0].max() , Vp[:,1].min() , Vp[:,1].max() 
        return minX , maxX, minY, maxY
        
    def pose_vertices(self,  pose):
        '''
          Compute the locations of the vertices of the pattern when the 
          pattern is in pose 'pose'.
          
          @return          
             Vp : vertices of the pattern when in pose 'pose'       
        '''
        theta,scale = pose[2:4]
        #print('theta {} scale {}'.format(theta, scale))
        T = pose[:2]   # translation vector 
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        Vp = (self.V-self.V[0]).dot(rot_mat.T)*scale + self.V[0]+ T 
        #print(Vp)
        return  Vp
        
#-----------------------------------------------------------------------------

class Square(Pattern):
    '''
    Create a Square pattern of side length 'side'
    '''
    def __init__(self):
        V = ((0,0),(1,0),(1,1),(0,1))
        E = ((0,1),(1,2),(2,3),(3,0))
        super().__init__(V,E)

#-----------------------------------------------------------------------------

class Triangle(Pattern):
    '''
    Create a Square pattern of side length 'side'
    '''
    def __init__(self, height):
        V = ((0,0),(1,0),(0.5,height))
        E = ((0,1),(1,2),(2,0))
        super().__init__(V,E)
        
#-----------------------------------------------------------------------------

def pat_image(pat_list, pose_list, imf=None):
    '''
    Create a black and white image by drawing patterns in the specified poses.
    '''
    if imf is None:
        imf = np.zeros(default_imf_shape,dtype=np.float32)
    #
    for pat, pos in zip(pat_list,pose_list):
        pat.draw_im(pos,imf)        
    #
    return imf

#-----------------------------------------------------------------------------

def dist_image(imf):
    '''
    Return the distance image 'imd' of 'imf'
    imd[r,c] is the distance  of pixel (r,c) to the closest edge pixel.
    '''
    imf_inv = cv2.bitwise_not(skimage.img_as_ubyte(imf)) 
    imd = cv2.distanceTransform(imf_inv ,cv2.DIST_L2,5)
#    plt.imshow(imf_inv)
#    plt.title('imf_inv')
#    plt.figure()
#    plt.imshow(imd)
#    plt.title('imd')
#    plt.colorbar()
    return imd

#-----------------------------------------------------------------------------

def make_test_image_1(show=False):
    ps = Square()
    pt = Triangle(2)
    
    pat_list = [ps, ps, pt, pt]
    pose_list = [ (10, 20, np.pi/6, 20), (50,30,0,30), 
                 (100,30, np.pi/3,40), (100,50, -np.pi/3,30)]    
#    region = (45,90,25,60)
    imf = pat_image(
                     pat_list ,
                     pose_list)

    imd = dist_image(imf)
    
    if show:
        plt.figure()
        plt.imshow(imf)
        plt.title('imf')
        plt.figure()
        plt.imshow(imd)
        plt.title('imd')
        plt.colorbar()
        plt.show()    
    return imf, imd, pat_list, pose_list
#-----------------------------------------------------------------------------

def make_test_image_2(show=False):
#    ps = Square()
    pt = Triangle(2)
    
    pat_list = [ pt]
    pose_list = [  (100,50, -np.pi/3,30)]
    
#    region = (45,90,25,60)
    imf = pat_image(
                     pat_list ,
                     pose_list)

    imd = dist_image(imf)
    
    if show:
        plt.figure()
        plt.imshow(imf)
        plt.title('imf')
        plt.figure()
        plt.imshow(imd)
        plt.title('imd')
        plt.colorbar()
        plt.show()    
    return imf, imd, pat_list, pose_list

#-----------------------------------------------------------------------------

def replay_search(pat_list, 
                  pose_list, 
                  pat,
                  L_search
                  ):
    '''
    Show how the search went
    '''        
    print('Close the figure to see the next generation!')
    for i in range(len(L_search)):
        # redraw everything at each iteration
        fig, ax = plt.subplots()
        # plot the targets in red
        for p, pose in zip(pat_list,pose_list):
            p.edge_color = [1,0,0] # draw the pattern in red
            p.draw_ax(ax,pose)        
        # plot the individuals in green
        W =  L_search[i]
        pat.edge_color = [0,1,0] # draw the pattern in red
        for pose in W:
            pat.draw_ax(ax,pose)       
        ax.set_xlim(0,200)
        ax.set_ylim(0,100)
        plt.title('Step {} out of {}'.format(i,len(L_search)))
        plt.show()    
        
#-----------------------------------------------------------------------------

def display_solution(pat_list, 
                  pose_list, 
                  pat,
                  pose_found
                  ):
    '''
    Show the solution found
    '''        
    # redraw everything at each iteration
    fig, ax = plt.subplots()
    # plot the targets in red
    for p, pose in zip(pat_list,pose_list):
        p.edge_color = [1,0,0] # draw the pattern in red
        p.draw_ax(ax,pose)        
    # plot the individuals in green
    pat.edge_color = [0,1,0] # draw the pattern in green
    pat.draw_ax(ax,pose_found)    
    ax.set_xlim(0,200)
    ax.set_ylim(0,100)
    
    plt.title('Found solution')
#    plt.show()    


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        

#def replay_search_v0(pat_list, 
#                  pose_list, 
#                  pat,
#                  L_search
#                  ):
#    '''
#    Show how the search went.
#    Buggy version!!
#    '''
#
#    def get_xydata(p,pose):
#        '''
#        
#        Compute the set of pixels of pattern 'p' in pose 'pose'
#        @return
#          x , y
#              where x[i], y[i] are the coords of the ith point in the pattern
#          
#        @param
#            p : pattern
#            pose : a pose
#        '''
#        Vp = p.pose_vertices(pose)
#        xdata,ydata = [],[]
#        for i,j in pat.E:
#            X,Y, _, _ = scan_segment(Vp[i], Vp[j])
#            xdata.extend(X)
#            ydata.extend(Y)
#        return xdata, ydata
#        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
#    fig, ax = plt.subplots()
#
#    for p, pose in zip(pat_list,pose_list):
#        p.draw_ax(ax,pose)        
#    
#    # initial pose 
#    pose = L_search[0][1]       
#    xdata, ydata = get_xydata(pat,pose)
#    line, = ax.plot(xdata,ydata,'r-')   
#    plt.title('Search Animation')
#    
#    def animate(i):
#        pose = L_search[i][1]       
#        xdata, ydata = get_xydata(pat,pose)
#        line.set_xdata(xdata)  # update the data
#        line.set_ydata(ydata)  # update the data
##        title_text.set_text(str(i))
##        title_text.stale = True
#        print( i, end='\n' if i%20==0 else ' ', flush=True)
#        return line,
#    
#    # Init only required for blitting to give a clean slate.
##    def init():
##        line.set_ydata(np.ma.array(x, mask=True))
##        return line,
#    
#    animation.FuncAnimation(fig, animate, np.arange(1, len(L_search)), #init_func=init,
#                                  interval=50, blit=True)
#    plt.show()    
##-----------------------------------------------------------------------------
#
#
#
#        # if any outside return np.inf
#        total_pix = 0      # total number of pixels scanned inside 'imf'
#        total_score = 0    # total score     
#        for i,j in self.E:
#            X, Y, S, all_inside = scan_segment(Vp[i], Vp[j], imf)
#            if not all_inside:
#                return np.inf, Vp
#            total_pix += len(X)
#            total_score += S.sum()
#        #
#        score =  total_score/total_pix    
#        return score, Vp
