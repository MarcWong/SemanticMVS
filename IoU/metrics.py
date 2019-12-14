import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    # print(cm)

    plt.figure(figsize=(12,9))
    plt.imshow(cm, cmap='Blues', aspect='auto', vmin=0, vmax=1)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上

    plt.yticks(num_local, labels_name, rotation='90')    # 将标签印在y轴坐标上
    plt.ylabel('Ground Truth label')
    plt.xlabel('Predicted label')

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
    def Save_Matrix(self):
        plot_confusion_matrix(self.confusion_matrix[1:,1:], ["building", "vegetarian", "car", "road"], "Confusion Matrix")
        plt.savefig('confusion_matrix.png', format='png')

        # plt.show()

    def Pixel_Accuracy(self):
        np.save('seg.npy', self.confusion_matrix[1:,1:])
        self.Save_Matrix()
        Acc = np.diag(self.confusion_matrix[1:,1:]).sum() / self.confusion_matrix[1:,1:].sum()
        # Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        # class Precision, Recall, F1
        n = self.confusion_matrix[1:,1:]
        m = n[0,0]
        a = np.delete(n,0,1)
        a = np.delete(a,0,0)
        accuracy = (m + a.sum()) / n.sum()
        recall = m / n[0].sum()
        precision = m / n[:,0].sum()
        f1 = 2 * precision * recall / (precision + recall)
        print(format(accuracy, '.4f'), format(precision, '.4f'), format(recall, '.4f'), format(f1, '.4f'))


        n = self.confusion_matrix[1:,1:]
        m = n[1,1]
        a = np.delete(n,1,1)
        a = np.delete(a,1,0)
        accuracy = (m + a.sum()) / n.sum()
        recall = m / n[1].sum()
        precision = m / n[:,1].sum()
        f1 = 2 * precision * recall / (precision + recall)
        print(format(accuracy, '.4f'), format(precision, '.4f'), format(recall, '.4f'), format(f1, '.4f'))

        n = self.confusion_matrix[1:,1:]
        m = n[2,2]
        a = np.delete(n,2,1)
        a = np.delete(a,2,0)
        accuracy = (m + a.sum()) / n.sum()
        recall = m / n[2].sum()
        precision = m / n[:,2].sum()
        f1 = 2 * precision * recall / (precision + recall)
        print(format(accuracy, '.4f'), format(precision, '.4f'), format(recall, '.4f'), format(f1, '.4f'))

        n = self.confusion_matrix[1:,1:]
        m = n[3,3]
        a = np.delete(n,3,1)
        a = np.delete(a,3,0)
        precision = (m + a.sum()) / n.sum()
        accuracy = m / n[3].sum()
        recall = m / n[:,3].sum()
        f1 = 2 * precision * recall / (precision + recall)
        print(format(accuracy, '.4f'), format(precision, '.4f'), format(recall, '.4f'), format(f1, '.4f'))

        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




