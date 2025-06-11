import base64
from PIL import Image
from io import BytesIO
import os

def base64_to_image(base64_string):
    '''
    This method converts a base64 string to an image
    :param base64_string:
    :return: Image in the form of bytes
    '''
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    return image_bytes

def create_image_from_bytes(image_bytes):
    '''
    This method creates an image from bytes
    :param image_bytes:
    :return: PIL image
    '''
    image_stream = BytesIO(image_bytes)

    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    return image

def show_image(image):
    '''
    This method displays an image
    :param image:
    :return: None
    '''
    image.show()

def save_image(image, path_to_save, filename):
    '''
    This method saves an image
    :param image:
    :param filename:
    :return:
    '''
    if(not os.path.exists(path_to_save)):
        os.mkdir(path_to_save)
    image.save(os.path.join(path_to_save, filename))
