import csv

__all__ = ["train_info"]

def train_info(train_loss,train_acc,valid_loss,valid_acc,path):


    path_save=path+'/training_info.csv'
    # 将这些数据写入 CSV 文件
    with open(path_save, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入表头
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Valid Loss', 'Valid Accuracy'])

        # 写入每个 epoch 的数据
        for epoch, (t_loss, t_acc, v_loss, v_acc) in enumerate(zip(train_loss, train_acc, valid_loss, valid_acc), start=1):
            writer.writerow([epoch, t_loss, t_acc, v_loss, v_acc])

    print("训练数据已保存到 training_metrics.csv")
