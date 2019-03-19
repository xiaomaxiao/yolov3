import numpy as np 



def voc_ap(rec,prec,use_07_metric = False):
    '''
    compute voc ap given precision and recall , 
    if use_07_metric is true ,use the voc 01 11-points method.
    Args:
        rec:Recall
        prec:Precision 
    retrun AP
    '''
    if(use_07_metric):
        ap = 0.
        for i in range(0.0,0.1,1.1):
            if(np.sum(rec>=i)==0):
                p =0
            else:
                p = np.max(rec>=i)
            ap = ap + p 
        ap = ap / 11.
    else:
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))

        for i in range(mrec.shape[0]-1,0,-1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])
        
        #因为rec pre 是按照每个bbox存的可能出现重复的recall，这里剔除掉
        i = np.where(mrec[1:]!=mrec[:-1])[0]

        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
        return ap

def voc_eval(gtboxes,bboxes,overthreshold=0.5):
    '''
    计算单个类别的mAP
    Args:
        gtboxes: the ground truth bounding boxes {'imagename':[[x1,x2,y1,y2],[.....]]}
        bboxes:the predict bounding boxes {'imagename':[[x1,x2,y1,y2,confidence],[.....]]}
        overthreshold:mIou threshold default =0.5
    return :percision recall  AP
    '''

    BB = []
    imgnames = []

    for k,v in bboxes.items():
        for b in v:
            BB.append(b)
            imgnames.append(k)

    #如果一个gtboxes 已经属于某个bboxes则标记为1，以后不参与IOU计算
    det= {k:[0] * len(v) for k,v in gtboxes.items()}
    

    BB = np.array(BB)
    imgnames = np.array(imgnames)

    #sort by confidence   
    confidence = BB[:,-1]
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind,:]
    imgnames = imgnames[sorted_ind]

    FP =  np.zeros(shape = (BB.shape[0]))
    TP = np.zeros_like(FP)

    for d in range(BB.shape[0]):
        #compute overlaps
        bb = BB[d]
        R = det[imgnames[d]]
        BBGT = gtboxes[imgnames[d]]
        BBGT = np.array(BBGT)
        ixmin = np.maximum(BBGT[:,0],bb[0])
        iymin = np.maximum(BBGT[:,1],bb[1])
        ixmax = np.minimum(BBGT[:,2],bb[2])
        iymax = np.minimum(BBGT[:,3],bb[3])
        iw = np.maximum(ixmax - ixmin + 1.0,0.)
        ih = np.maximum(iymax - iymin + 1.0,0.)
        intersection = iw * ih 

        union =(bb[2]-bb[0]+1.0) * (bb[3]-bb[1]+1.0) + \
            (BBGT[:,2]-BBGT[:,0]+1) * (BBGT[:,3] -BBGT[:,1]+1) - \
            intersection

        iou = intersection / union 

        overmax = np.max(iou)
        jmax = np.argmax(iou)

        if(overmax> overthreshold and  R[jmax]==0):
            TP[d] = 1
            R[jmax] = 1
        else:
            FP[d] = 1
        
    #computer precision and recall
    FP = np.cumsum(FP)
    TP = np.cumsum(TP)
    prec = TP / np.maximum(TP + FP,0.01)

    num_gtboxes = 0
    for k,v in gtboxes.items():
        num_gtboxes +=len(v)
    rec = TP / np.maximum(num_gtboxes ,0.01)
    ap = voc_ap(rec,prec)
    return prec, rec , ap 




#test 
def create_test_data():
    gt_data = {}
    gt_data['img1'] = []
    gt_data['img1'].append([25,16,38,56])
    gt_data['img1'].append([129,123,41,62])

    gt_data['img2'] = []
    gt_data['img2'].append([123,11,43,55])
    gt_data['img2'].append([38,132,59,45])

    gt_data['img3'] = []
    gt_data['img3'].append([16,14,35,48])
    gt_data['img3'].append([123,30,49,44])
    gt_data['img3'].append([99,139,47,47])

    gt_data['img4'] = []
    gt_data['img4'].append([53,42,40,52])
    gt_data['img4'].append([154,43,31,34])

    gt_data['img5'] = []
    gt_data['img5'].append([59,31,44,51])
    gt_data['img5'].append([48,128,34,52])

    gt_data['img6'] = []
    gt_data['img6'].append([36,89,52,76])
    gt_data['img6'].append([62,58,44,67])

    gt_data['img7'] = []
    gt_data['img7'].append([28,31,55,63])
    gt_data['img7'].append([58,67,50,58])

    pre_data = {}
    pre_data['img1'] =[]
    pre_data['img1'].append([5,67,31,48,.88])
    pre_data['img1'].append([119,111,40,67,.70])
    pre_data['img1'].append([124,9,49,67,.80])

    pre_data['img2'] =[]
    pre_data['img2'].append([64,111,64,58,.71])
    pre_data['img2'].append([26,140,60,47,.54])
    pre_data['img2'].append([19,18,43,35,.74])

    pre_data['img3'] =[]
    pre_data['img3'].append([109,15,77,39,.18])
    pre_data['img3'].append([86,63,46,45,.67])
    pre_data['img3'].append([160,62,36,53,.38])
    pre_data['img3'].append([105,131,47,47,.91])
    pre_data['img3'].append([18,148,40,44,.44])

    pre_data['img4'] =[]
    pre_data['img4'].append([83,28,28,26,.35])
    pre_data['img4'].append([28,68,42,67,.78])
    pre_data['img4'].append([87,89,25,39,.45])
    pre_data['img4'].append([10,155,60,26,.14])

    pre_data['img5'] =[]
    pre_data['img5'].append([50,38,28,46,.62])
    pre_data['img5'].append([95,11,53,28,.44])
    pre_data['img5'].append([29,131,72,29,.95])
    pre_data['img5'].append([29,163,72,29,.23])

    pre_data['img6'] =[]
    pre_data['img6'].append([43,48,74,38,.45])
    pre_data['img6'].append([17,155,29,35,.84])
    pre_data['img6'].append([95,110,25,42,.43])

    pre_data['img7'] =[]
    pre_data['img7'].append([16,20,101,88,.48])
    pre_data['img7'].append([33,116,37,49,.95])

    #format xywh  to x1y1x2y2

    for key in gt_data.keys():
        gt1 = gt_data[key]
        pre1= pre_data[key]

        for gt in gt1 :
            gt[2] = gt[0] + gt[2]
            gt[3] = gt[1] + gt[3]
        
        for pre in pre1:
            pre[2] = pre[0] + pre[2]
            pre[3] = pre[1] + pre[3]
        
    return gt_data,pre_data


gt_data,pre_data = create_test_data()

prec,rec,ap = voc_eval(gt_data,pre_data,overthreshold=0.3)