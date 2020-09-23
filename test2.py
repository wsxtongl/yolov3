from PIL import Image
import matplotlib.pyplot as plt
def Square_Generated (read_file): # 创建一个函数用来产生所需要的正方形图片转化
    image = Image.open(read_file)   # 导入图片
    w, h = image.size  # 得到图片的大小

    new_image = Image.new('RGB', size=(max(w, h), max(w, h)),color= 'black')  # 创建新的一个图片，大小取长款中最长的一边，color决定了图片中填充的颜色

    length = int(abs(w - h)/2)  # 一侧需要填充的长度
    if w < h:
        box = (length, 0)
        new_image.paste(image, box)       #产生新的图片

    else:
        box =(0, length)
        new_image.paste(image, box)

    return new_image


if __name__ == '__main__':
    img = Square_Generated("2007_000129.jpg")
    plt.imshow(img)
    plt.show()