import pandas as pd

from model import *


def show_images(images, annotations=None, classes_dict=None, figsize=(12, 12), ncols=2, show_labels=True):
    """
    Shows a sample of images with annotations for them.

    Args:
        images (list): Sample of images.
        annotations (list, optional): Annotations for pictures.
        classes_dict (dict): Dictionary with names of the classes and their labels.
        figsize (tuple, optional): Size of the plot.
        ncols (int, optional): Number of images in one line.
        show_labels (bool, optional): Show labels or not.
    """

    n = len(images)
    nrows = math.ceil(n/ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        image, annotation = images[i].permute(1, 2, 0).detach(), annotations[i]
        bboxes, labels = annotation['boxes'].detach(), annotation['labels'].detach()
        for bbox, label in zip(bboxes, labels):
            x, y = bbox[0].item(), bbox[1].item()
            width, height = bbox[2].item() - bbox[0].item(), bbox[3].item() - bbox[1].item()
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if show_labels:
                if classes_dict is None:
                    raise 'You should specified dictionary with classes names and labels.'
                class_name = str([i for i in classes_dict if classes_dict[i] == label][0])
                lx = x + width / 2.0
                ly = y + height / 8.0
                ax.annotate(class_name, (lx, ly), fontsize=6, fontweight="bold", color="red", ha='center', va='center')
        ax.imshow(image)
    plt.tight_layout()
    plt.show()


def evaluate_model(trained_model, dataset, classes_dict=CLASSES_DICT):
    """Creates dataframes with evaluation metrics values."""

    images = [dataset[i][0] for i in range(len(dataset))]
    targets = [dataset[i][1] for i in range(len(dataset))]
    predictions = trained_model(images)
    class_name_l, ap_50_l, ap_75_l, ap_95_l, mean_ap_l, acc_l, red_l = [], [], [], [], [], [], []
    for item in classes_dict.items():
        class_name, label = item[0], item[1]
        ap_50 = precision_recall_ap(targets, predictions, iou_thresh=0.5, class_label=label)[2]
        ap_75 = precision_recall_ap(targets, predictions, iou_thresh=0.75, class_label=label)[2]
        ap_95 = precision_recall_ap(targets, predictions, iou_thresh=0.95, class_label=label)[2]
        mean_ap = mean_average_precision(targets, predictions, class_label=label)
        acc = accuracy(targets, predictions, class_label=label)
        red = redundancy(targets, predictions, class_label=label)
        class_name_l.append(class_name)
        ap_50_l.append(ap_50)
        ap_75_l.append(ap_75)
        ap_95_l.append(ap_95)
        mean_ap_l.append(mean_ap)
        acc_l.append(acc)
        red_l.append(red)
    df1 = pd.DataFrame.from_dict({'class': class_name_l,
                                  'AP50': ap_50_l,
                                  'AP75': ap_75_l,
                                  'AP95': ap_95_l,
                                  'mAP': mean_ap_l,
                                  'accuracy': acc_l,
                                  'redundancy': red_l
                                  })
    ap_50 = precision_recall_ap(targets, predictions, iou_thresh=0.5)[2]
    ap_75 = precision_recall_ap(targets, predictions, iou_thresh=0.75)[2]
    ap_95 = precision_recall_ap(targets, predictions, iou_thresh=0.95)[2]
    mean_ap = mean_average_precision(targets, predictions)
    acc = accuracy(targets, predictions)
    red = redundancy(targets, predictions)
    df2 = pd.DataFrame.from_dict({'AP50': [ap_50],
                                  'AP75': [ap_75],
                                  'AP95': [ap_95],
                                  'mAP': [mean_ap],
                                  'accuracy': [acc],
                                  'redundancy': [red]
                                  })
    return df1, df2


if __name__ == '__main__':
    if not IS_TRAIN:
        detector = FasterRCNN()
        trainer = Trainer(max_epochs=MAX_EPOCHS, fast_dev_run=False)
        trainer.fit(detector)

    # model = FasterRCNN.load_from_checkpoint('lightning_logs/version_6/checkpoints/epoch=3-step=204.ckpt')
    # test_dataset = CPDDataset(TEST_DIR, CLASSES_DICT)
    # random_indices = np.random.randint(0, high=20, size=4)
    # images = [test_dataset[i][0] for i in random_indices]
    # annotations = [test_dataset[i][1] for i in random_indices]
    # predictions = model(images)
    # # print(precision_recall_ap(annotations, predictions))
    # # show_images(images, annotations=predictions, classes_dict=CLASSES_DICT)
    # # print(accuracy(annotations, predictions))
    # df_classes, df_all = evaluate_model(model, test_dataset)
    # display(df_classes)
    # display(df_all)



