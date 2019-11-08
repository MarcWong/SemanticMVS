from metrics import Evaluator
from PIL import Image
import numpy as np

def evaluate_single(gt,pred,num_of_class):
    evaluator = Evaluator(num_of_class)
    # evaluator.reset()
    evaluator.add_batch(gt,pred)
    
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    return Acc, Acc_class, mIoU, FWIoU

def evaluate_batch(gt_list,pred_list,num_of_class):
    evaluator = Evaluator(num_of_class)
    # evaluator.reset()
    for i in range(len(gt_list)):
        evaluator.add_batch(gt_list[i],pred_list[i])
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    return Acc, Acc_class, mIoU, FWIoU

def main():
    pred = np.array(Image.open("c.png"))
    gt = np.array(Image.open("d.png"))
    print(evaluate_single(pred,gt,5))

    pred_list = []
    gt_list = []
    pred_list.append(np.array(Image.open("c.png")))
    pred_list.append(np.array(Image.open("d.png")))
    gt_list.append(np.array(Image.open("c.png")))
    gt_list.append(np.array(Image.open("d.png")))
    print(evaluate_batch(gt_list,pred_list,5))

main()