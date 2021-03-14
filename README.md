## Bird Classfication
  <div align=center><img width="650" src="/CSE455_Project/bird.jpg"/></div>
### Problem description
We were doing image recognition of vehicles and using the data from Kaggle and Cifar. This recognition includes convolutional neural networks in pytorch. However, once the professor announced the bird classification challenge, we understood that we can use the same skill for classifying different types of birds. We can use a neural network to train the bird dataset with up to 38562 images in it and see how accurate our model is. In addition to our accuracy, we also can compete with others training accuracy in Kaggle. This can help us to think about how to improve our model, including changing the number of epochs or the size of learning rate.   
Here is the link of our Github Repo: [Github website](https://github.com/JingC123/CSE455_Project)

### Related work

**ResNet**: PyTorch's ResNet model was used to be our pretrained model. Resnet models were proposed in “Deep Residual Learning for Image Recognition”. There are several versions of resnet models which contain different layers respectively. In the kaggle competition, “resnet18”(vision:v0.6.0) which contains 18 layers, was used as an example of the pretrained model. Detailed model architectures and accuracy can be found online. We tested different versions of resnet pretrained model to get the best accuracy result for the competition problem.  

**Dataset**: We are using the kaggle bird identification dataset provided by the instructor
There are 555 different images categories (birds) given by integers [0-554].
[Birds](https://www.kaggle.com/c/birds21wi)  
### Methodology

**Platform and tool**:
We decide to use colab as our developing platform since it is more convenient for a team project. Colab also provides online GPUs computation from Google. We can use those GPUs with cuda computation for training models and accelerating computational speed. In addition to the online platform, the programming tool we used in this project is Pytorch. Pytorch supports several computational functions in the neural network. We can design our own neural network and train the model in Pytorch. It also supports cuda computation in GPU. We can use the GPU in colab with Pytorch to train the model and accelerate computation speed. 
  
**Data preprocessing**:
After having a preliminary check on the dataset, we found some place that could be improved. First, we are mostly training our model using 128*128 sized image, however, the size of the picture could be much bigger, thus, we could preprocess our image so that the data loader does not have to resize our image every time. Second, the size of the dataset is large but not too large that cannot be stored in the ram of the colab. Thus, we wrote a custom dataset class that inherited torch.util.data.Dataset. When the custom dataset is created, it stores all data into a dictionary. In this case, all data is loaded into Ram and the training process will not have to load data from disk. By doing this, our run time for training reduced from 2 hr to 30 minute.

**Overfitting**:
We use the pretrained Resnet50 net as our model. In the first 10 epochs, the performance of the net is very good. The training and testing accuracy we got are 0.73, 0.58.  
But unfortunately, when we trained the model to 20 epochs, we encountered a very serious overfitting problem. The training accuracy we got is 0.98, but the testing accuracy we got is only 0.72. And when we kept training this model, the situation was not going well -the testing accuracy almost did not increase at all.  
We have adopted the following methods to solve this problem.
* Add decay  
As we know, we can add decay to prevent overfitting. The default decay is 0.0005. We set the decay to 0.005. But the improvement effect is minimal. The testing accuracy rose from 0.72 to 0.73.
* Add dropout layer  
We try to add a dropout layer after the convolutional layer to prevent overfitting.
* Use another pretrianed net  
We try to use another pretrained net - Resnet152.


### Results
We trained the model and got the generator loss near 3, the generator identity loss near 0.5, the cycle loss near 1, and discriminator loss near 0.2. The GAN loss is increasing during the training, while the discriminator loss shows a linear reduction and all other losses shows an exponential reduction.
We used the trained model to generate several testing pictures. Some of them showed good results of age shifting, which is described in the examples section.

### Expectation
We want try some other pretrained models and improve the predict method to get a better test result in the future work.



### Codes


