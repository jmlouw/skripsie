import numpy as np

class Gaussian(object):

    def __init__(self, RVs, mean = None, cov = None,  K = None, h = None):
            
        self.RVs = RVs
        
        if mean is not None and cov is not None:
            self.mean = mean
            self.cov = cov
            
            self.updateCan()                      

        elif K is not None and h is not None:
            self.K = K
            self.h = h
            
            
            #self.updateCov()
                   

##update canonical parameters##
    def updateCan(self):
        self.K = np.linalg.inv(self.cov)
        self.h = np.dot(self.K, self.mean)
        #self.g = -0.5*np.transpose(self.mean).dot(self.h)+np.log(norm)

##update covariance parameters##
    def updateCov(self):
        self.cov = np.linalg.inv(self.K)
        self.mean = self.cov.dot(self.h)
        

##re-arrange scope##
    def swopEntries(self, index1, index2):
       
        K = self.K
        h = self.h
        
        K[:,[index1, index2]] = K[:,[index2, index1]]
        K[[index1, index2],:] = K[[index2, index1],:]
        
        temp = h[index1]
        h[index1] = h[index2]        
        h[index2] = temp
    
    def reArrangeEntries(self, newOrder):
       
        K = self.K
        h = self.h
        RVs = self.RVs 
        
        oldOrder = np.arange(len(K))
        
        self.K[:,[oldOrder]] = K[:,[newOrder]]
        self.K[[oldOrder],:] = K[[newOrder],:]
        self.h[oldOrder] = h[newOrder]
        
        self.RVs = [RVs[i] for i in newOrder]
       
        
        
##extend scope##
    def extendScope(self, rv):
        K = self.K
        h = self.h
                
        self.K = np.pad(K, ((0,1),(0,1)), mode = 'constant')
        self.h = np.pad(h, (0,1), mode = 'constant')
        
        self.RVs.append(rv)        

#marginalize##
    def marginalizeUpdate(self, lenKxx, lenKyy):       
        KxxStart = len(self.K) - lenKxx - lenKyy        
        Kxx = self.K[KxxStart:lenKxx,KxxStart:lenKxx]
                        
        KyyStart = len(self.K) - lenKyy
        KyyEnd = len(self.K)
        
        Kyy = self.K[KyyStart: KyyEnd, KyyStart: KyyEnd]
        Kxy = self.K[KxxStart: lenKxx, KyyStart: KyyEnd]
        Kyx = self.K[KyyStart: KyyEnd, KxxStart: lenKxx]
        
        hx = self.h[KxxStart: lenKxx]
        hy = self.h[KyyStart: KyyEnd]           
           
        self.K = Kxx - Kxy.dot(np.linalg.inv(Kyy).dot(Kyx))
        self.K = hx - Kxy.dot(np.linalg.inv(Kyy)).dot(hy)
        #g_ac =  self.g + 0.5*(np.log(2*np.pi*np.linalg.inv(Kyy))+(hy.T).dot(np.linalg.inv(Kyy)).dot(hy))
       # self.g = g_ac[0][0]
        
    def marginalizeIndicesUpdate(self, arrIndices):##recieves indices to integrate over(marginalize out)
        
        oldOrder = np.arange(len(self.K))
        newOrder = np.concatenate((np.delete(oldOrder, arrIndices), arrIndices))
        
        self.reArrangeEntries(newOrder)
        
        KyyStart = len(self.K) - len(arrIndices)        
        
        Kxx = self.K[: KyyStart, : KyyStart]
        Kyy = self.K[KyyStart: , KyyStart: ]
        Kxy = self.K[: KyyStart, KyyStart: ]
        Kyx = self.K[KyyStart: , : KyyStart]
        
        hx = self.h[: KyyStart]
        hy = self.h[KyyStart: ]
        
        self.K = Kxx - Kxy.dot(np.linalg.inv(Kyy).dot(Kyx))
        self.h = hx - Kxy.dot(np.linalg.inv(Kyy)).dot(hy)
        
        for i in sorted(arrIndices, reverse=True): 
            del self.RVs[i]
       # g_ac =  self.g + 0.5*(np.log(2*np.pi*np.linalg.inv(Kyy))+(hy.T).dot(np.linalg.inv(Kyy)).dot(hy))
       # self.g = g_ac[0][0]
       
      
    
    def evidenceUpdate(self, arrIndices, evidence_y): 
        oldOrder = np.arange(len(self.K))
        newOrder = np.concatenate((np.delete(oldOrder, arrIndices), arrIndices))

        self.reArrangeEntries(newOrder)
        
        KyyStart = len(self.K) - len(arrIndices)  
        Kxx = self.K[: KyyStart, : KyyStart]
        Kxy = self.K[: KyyStart, KyyStart: ]
        
        hx = self.h[: KyyStart]
        
        self.K = Kxx
        self.h = hx - Kxy.dot(evidence_y)
        
        delArr = oldOrder[len(oldOrder)- len(arrIndices):]      

        for i in sorted(delArr, reverse=True): 
            del self.RVs[i]

       # self.g = self.g + (hy.T).dot(evidence_y) - 0.5*(evidence_y.T).dot(Kyy).dot(evidence_y)      
        
                
##multiply##         
    def multiplyUpdate(self, can):
        self.K = self.K + can.K
        self.h = self.h + can.h
        #self.g = self.g + can.g
        
        self.updateCov()
        
        print("scopes do not match")
        
    def multiplyNew(self, can):
        newK = self.K + can.K
        newh = self.h + can.h
        #newg = self.g + can.g       
        
        return Gaussian(RVs = self.RVs, K = newK, h = newh)

        
    def normalCoef(self): ##determines the normalisation coefficient
        self.updateCov()
        K_len = len(self.cov)
        cov_det = np.linalg.det(self.cov)
        
        coef = ((2*np.pi)**(K_len/2))*((cov_det)**0.5)
        
        return (1/coef)
        
    def toStringCan(self):
          print(self.RVs)
          
          print('K\n',self.K)
          print('h\n',self.h)
         # print('g\n',self.g)
         
    def toStringCov(self):
        
        print(self.RVs)        
        print('Mean\n', self.mean)
        print('Sigma\n', self.cov)


     
          

