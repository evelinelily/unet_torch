from skimage.io import imread
from skimage.transform import resize
import albumentations as A
import matplotlib.pyplot as plt


def round_clip_0_1(x,**kwargs):
    return x.round().clip(0, 1)


def brightness_adjustment(x,**kwargs):

    x_max = np.max(x)
    x_min = np.min(x)

    r = (np.random.random(1)-0.5)/5
    x = x*(1+r)
    x = np.clip(x, x_min, x_max)
    return x


# define heavy augmentations
def get_training_augmentation(imgsize):
    IMG_HEIGHT, IMG_WIDTH = imgsize
    train_transform = [
        A.HorizontalFlip(p=0.5),  #水平翻转
        A.VerticalFlip(p=0.5),
        A.IAAFliplr(p=0.5),
        #A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        A.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True, p=0.1),
        A.PadIfNeeded(IMG_HEIGHT, IMG_WIDTH),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Lambda(brightness_adjustment),
            ],
            p=0.5,),

        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)