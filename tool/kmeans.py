import numpy as np 

class yolo_kmeans:

    def __init__(self):
        """
        # Arguments:
            boxes: tuple or array (N,4),  [x1,y1,x2,y2]
            cluster_number : 聚类数量
        """
        pass
        #self.cluster_number = cluster_number
        #self.boxes = boxes

    def _iou(self,box,clusters):
        """
        calculates the intersection over union(IOU) between one box and k clusters
        # Arguments:
            boxes: tuple or array ,shifted to the origin ( width,height)
            clusters : tuple or array(k,2) where k is the number of clusters
        return : numpy array of shape(k,) 
        """
        x_min = np.minimum(clusters[:,0],box[0])
        y_min = np.minimun(clusters[:,1],box[1])
        
        if(np.count_nonzero(x_min==0) > 0 or np.count_nonzero(y_min==0)>0 ):
            raise ValueError('box has no area')

        intersection = x * y 
        box_area = box[0] * box[1]
        clusters_area = clusters[:,0] * cluseters[:,1]
        iou = intersection / (box_area + clusters_area - intersection)
        return iou 

    def _translate_boxes(self,boxes):
        """
        translater all boxes to the origin.(x1,y1,x2,y2)->(widht,height)
        # Arguments:
            boxes: numpy array of shape(n,4)
        return :numpy arrray of shape(n,2)
        """
        shift_boxes = np.zeros((boxes.shape[0],2))
        shift_boxes[:,0] = np.abs(boxes[:,0] - boxes[:,2])
        shift_boxes[:,1] = np.abs(boxes[:,1] - boxes[:,3])
        return shift_boxes

    def _avg_iou(self):
        pass


    def kmeans(self,boxes,k,dist = np.median):
        """
        用IOU指标 Kmeans聚类
        # Arguments:
           boxes: (n,4) [x1,y1,x2,y2]
           k : 聚类数量
           dist: 计算距离的函数
        return : clusters (k,2)
        """
        #坐标转换
        shift_boxes = self._translate_boxes(boxes)

        n = shift_boxes.shape[0]

        #存储每个box对应k个候选box的距离
        distance = np.empty((n,k))

        #随机选K个box作为种子
        np.random.seed()

        clusters = shift_boxes[np.random.choice(n,k,replace=True)]

        last_clusters = np.zeros((n,))
        while True:
            for i in range(n):
                distance[i] = 1 - self._iou(shift_boxes[i],clusters)
            #每个box属于 距离最小的 那个类
            nearest_clusters = np.argmin(distance,axis=-1) 
            if(last_clusters == nearest_clusters).all():
                break
           
            # 更新候选box
            for cl in range(k):
                clusters[cl] = dist(boxes[nearest_clusters==cl],axis=0)

            last_clusters = nearest_clusters

        return clusters
        