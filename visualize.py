#!/usr/bin/env python
# coding: utf-8

# In[26]:


import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[23]:


with open('performance.json') as f:
    performance = json.load(f)

IMAGE_DIR = "imgs"


# In[39]:


def visualize_loss(modelstr):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(performance[modelstr]["clf_train_losses"],
            label="Train Loss")
    ax1.plot(performance[modelstr]["clf_test_losses"],
            label="Test Loss")
    ax2.plot(performance[modelstr]["reg_train_losses"],
            label="Train Loss")
    ax2.plot(performance[modelstr]["reg_test_losses"],
            label="Test Loss")

    decay_epochs = [15, 30]
    y_lim1 = ax1.get_ylim()
    y_lim2 = ax2.get_ylim()
    for decay_epoch in decay_epochs:
        if decay_epoch == decay_epochs[-1]:
            ax1.plot([decay_epoch - 1, decay_epoch - 1],
                    y_lim1, color="purple", label="lr decay")
            ax2.plot([decay_epoch - 1, decay_epoch - 1], y_lim2,
                    color="purple", label="lr decay")
        else:
            ax1.plot([decay_epoch - 1, decay_epoch - 1], y_lim1, color="purple")
            ax2.plot([decay_epoch - 1, decay_epoch - 1], y_lim2, color="purple")

    ax1.set_xticks(np.arange(4, 40, 5))
    ax1.set_xticklabels(np.arange(5, 41, 5))
    ax1.set_title("Classification Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2.set_title("Regression Loss")
    ax2.set_xticks(np.arange(4, 40, 5))
    ax2.set_xticklabels(np.arange(5, 41, 5))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")

    base, init_lr, pretrained = modelstr.split(sep="_")

    fig.suptitle(f"Base Model = ResNet{base}, Initial Learning Rate = {init_lr}, Pretrained = {pretrained}")

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, modelstr + "_loss.png"), facecolor="white")


# In[40]:


for base in ["18", "34", "50"]:
    for init_lr in ["1.0", "0.1", "0.01"]:
        for pretrained in ["True", "False"]:
            modelstr = "_".join([base, init_lr, pretrained])
            visualize_loss(modelstr)


# In[29]:


modelstrs = []
last_clf_train_losses = []
last_clf_test_losses = []
last_reg_train_losses = []
last_reg_test_losses = []

for base in ["18", "34", "50"]:
    for init_lr in ["1.0", "0.1", "0.01"]:
        for pretrained in ["True", "False"]:
            modelstr = "_".join([base, init_lr, pretrained])
            modelstrs.append(modelstr)
            last_clf_train_losses.append(
                performance[modelstr]["clf_train_losses"][-1])
            last_clf_test_losses.append(
                performance[modelstr]["clf_test_losses"][-1])
            last_reg_train_losses.append(
                performance[modelstr]["reg_train_losses"][-1])
            last_reg_test_losses.append(
                performance[modelstr]["reg_test_losses"][-1])

performance_df = pd.DataFrame(
    np.array([modelstrs, last_clf_train_losses, last_clf_test_losses, last_reg_train_losses, last_reg_test_losses]).T, columns=["Model", "Classification Loss (Train)", "Classification Loss (Test)", "Regression Loss (Train)", "Regression Loss (Test)"])

performance_df


# In[42]:


performance_df.sort_values(by=["Classification Loss (Test)"])[["Model", "Classification Loss (Test)"]]


# In[43]:


performance_df.sort_values(by=["Regression Loss (Test)"])[["Model", "Regression Loss (Test)"]]


# In[35]:


min_clf_test = performance_df.sort_values(
    by=["Classification Loss (Test)"]).iloc[0]["Model"]
min_reg_test = performance_df.sort_values(
    by=["Regression Loss (Test)"]).iloc[0]["Model"]


# In[36]:


print("Model with Minimum Test Classification Loss: ", min_clf_test)
print("Model with Minimum Test Rergession Loss: ", min_reg_test)

