import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, box, is_auth=1):
    edgecolor = 'b' if is_auth == 0 else 'r'
    ex1 = box[0]
    ey1 = box[1]
    ex2 = box[2]
    ey2 = box[3]
    rect = patches.Rectangle((ex1,ey1),abs(ex1 - ex2),abs(ey1 - ey2),linewidth=1, edgecolor=edgecolor, fill=False)
    ax.add_patch(rect)

def vis(sample):
    ''' Assumes batch size of 4'''
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(sample[0])
    axs[0,1].imshow(sample[1])
    axs[1,0].imshow(sample[2])
    axs[1,1].imshow(sample[3])
    plt.show()

def visDet(sample, boxes, preds=None):
    ''' Assumes batch size of 4'''
    sample = sample.permute(0,2,3,1)
    fig, axs = plt.subplots(2,2)

    axs[0,0].imshow(sample[0])
    
    if preds is not None:
        for bbox, pred in zip(boxes[0], preds[0]):
            draw_box(axs[0,0], bbox, pred)
    else:
        for bbox in boxes[0]:
            draw_box(axs[0,0], bbox)

    axs[0,1].imshow(sample[1])
    
    if preds is not None:
        for bbox, pred in zip(boxes[1], preds[1]):
            draw_box(axs[0,1], bbox, pred)
    else:
        for bbox in boxes[1]:
            draw_box(axs[0,1], bbox)

    axs[1,0].imshow(sample[2])
    
    if preds is not None:
        for bbox, pred in zip(boxes[2], preds[2]):
            draw_box(axs[1,0], bbox, pred)
    else:
        for bbox in boxes[2]:
            draw_box(axs[1,0], bbox)

        
    axs[1,1].imshow(sample[3])
    
    if preds is not None:
        for bbox, pred in zip(boxes[3], preds[3]):
            draw_box(axs[1,1], bbox, pred)
    else:
        for bbox in boxes[3]:
            draw_box(axs[1,1], bbox)

    plt.show()