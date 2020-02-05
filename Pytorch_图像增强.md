### 1. 前言

- 数据增强对深度神经网络的训练来说是非常重要的，尤其是在数据量较小的情况下能起到扩充数据的效果。
- 本文总结了pytorch中使用**torchvision提供的transform模块中进行数据增强常用的7种方式。**

### 2. 图像增强

#### 2.1 PIL读取image

- pytorch提供的torchvision主要使用PIL的Image类进行处理，所以它数据增强函数大多数都是以PIL作为输入，并且以PIL作为输出。**将图片读取为PIL.Image类对象：**

```python
def read_PIL(image_path):
    image = Image.open(image_path)
    return image
```

- 这个函数是用PIL打开图片的常用操作。**后面的12种操作函数的输入都是read_PIL()返回的image对象。**选取PIL库是因为这是python的原生库，兼容性比较强。

#### 2.2 中心裁剪

- 中心裁剪的**是从图片的中心开始，裁剪出其四周指定长度和宽度的图片**，也就是获取原图的中心部分。
- 核心类是`transforms.CenterCrop(size)`，其中size参数是表示我们希望裁剪的尺寸的大小：

```python
def center_crop(image):
    CenterCrop = transforms.CenterCrop(size=(200, 200))
    cropped_image= CenterCrop(image)
    return cropped_image
```

- 例如下图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/pytorch/1-1.jpg" style="zoom:50%;" />

#### 2.3 随机裁剪

- 中心裁剪需要围绕图片中心点进行，但是**随机裁剪的结果来自图片的哪个部分是随机的。**
- 所使用的核心函数为`transforms.RandomCrop(size)`，其中size表示自己希望获得裁剪后图片的尺寸：

```python
def random_crop(image):
    RandomCrop = transforms.RandomCrop(size=(200, 200))
    random_image = RandomCrop(image)
    return random_image
```

- 例如下图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/pytorch/1-2.jpg" style="zoom:50%;" />

#### 2.4 Resize图像

- 图像的resize是最为常见的操作，尤其是当**手中数据尺寸和网络结构不匹配时，**比如网络只有3个池化层的进行降低特征图尺寸，但是手里的数据为1024x1024时，就需要降低图像的长和宽。
- 核心函数为`transforms.Resize(size)`，其中size依旧表示目标图像尺寸：

```python
def resize(image):
    Resize = transforms.Resize(size=(100, 50))
    resized_image = Resize(image)
    return resized_image
```

- 例如下图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/pytorch/1-3.jpg" style="zoom:50%;" />

#### 2.5 随机长宽比裁剪

- 随机长宽比裁剪的实现借助于`transforms.RandomResizedCrop`类，可以看出这个功能是Resize和Crop的随机组合，这在Inception网络的训练中比较有用。
- 这个类的初始化包含**3个参数（size， scale， ratio），size参数为目标图片的尺寸，其中scale参数代表输出图片占原始图片的百分比区间，ratio表示长宽比的取值区间，随机的意思就是在这两个区间中随机选取两个参数值：**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/pytorch/1-4.jpg" style="zoom:50%;" />

#### 2.6 随机水平翻转

- 随机水平翻转意思是有一半的可能性翻转，也有一半的可能性不翻转。
- 使用的是`transforms.RandomHorizontalFlip()`类：

```python
def horizontal_flip(image):
    HF = transforms.RandomHorizontalFlip()
    hf_image = HF(image)
    return hf_image
```

- 例如下图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/pytorch/1-5.jpg" style="zoom:50%;" />

#### 2.7 随机旋转

- 随机旋转仍然是图像像素位置变换的一种常用操作，我们可以设定自己希望旋转的角度区间
- `transforms.RandomRotation(degrees)`中的degrees参数表示旋转角度的选择范围：

```python
def random_rotation(image):
    RR = transforms.RandomRotation(degrees=(10, 80))
    rr_image = RR(image)
    return rr_image
```

- 下面展示了随机旋转前后的区别：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/pytorch/1-6.jpg" style="zoom:50%;" />