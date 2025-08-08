import csv

__all__ = ["train_info"]

def train_info(train_loss,train_acc,valid_loss,valid_acc,path):


    path_save=path+'/training_info.csv'
    # Make data to csv file
    with open(path_save, mode='w', newline='') as file:
        writer = csv.writer(file)

        #Write into the table header
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Valid Loss', 'Valid Accuracy'])

        # write the data of every epoch
        for epoch, (t_loss, t_acc, v_loss, v_acc) in enumerate(zip(train_loss, train_acc, valid_loss, valid_acc), start=1):
            writer.writerow([epoch, t_loss, t_acc, v_loss, v_acc])

    print("train_data are saved in training_metrics.csv")
