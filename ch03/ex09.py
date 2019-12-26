"""
PIL 패키지와 numpy 패키지를 이용하면,
이미지 파일(jpg, png, bmp, ...)의 픽셀 정보를 numpy.ndarray 형식으로 변환하거나
numpy.ndarray 형식의 이미지 픽셀 정보를 이미지 파일로 저장할 수 있습니다.
"""
from PIL import Image
import numpy as np

def image_to_pixel(image_file):
    """이미지 파일 이름(경로)를 파라미터로 전달받아서,
    numpy.ndarray에 픽셀 정보를 저장해서 리턴."""
    img = Image.open(image_file, mode = 'r') #fp = file pointer = file name
    print(type(img)) #ImageFile
    pixels = np.array(img) # 이미지 파일 객체를 numpy.ndarray 형식으로 변환
    print('pixels shape', pixels.shape) #(height, width, color)
    # color: 8bit(grey scale) (= 1byte = 1), 24bit (RGB)(= 3 byte = 3), 32bit(RGBA) ( = 4 byte = 4)
    # RGBA -> 0 이 투명 or 연한색, 255가 불투명 or 진한색
    return pixels



def pixel_to_image(pixel, image_file):
    """numpy.ndarray 형식의 이미지 픽셀 정보와, 저장할 파일 이름을 파라미터로
    전달받아서, 이미지 파일을 저장"""
    img = Image.fromarray(pixel) #ndarray (배열) 타입의 데이터를 이미지로 변환
    print(type(img)) # Image클래스
    img.show() # image viewer을 사용해서 이미지 보기
    img.save(image_file) #파일 이름을 파라미터로 줘서 저장 # 이미지 객체를 파일로 저장



if __name__ == '__main__':
    # image_to_pixel(), pixel_to_image() 함수 테스트
    pixels_1 = image_to_pixel('pengsoo.jpg')
    # <class 'PIL.JpegImagePlugin.JpegImageFile'> # 파일 클래스
    # pixels shape(1280, 720, 3) # width x height / 24bit color (3*8 = 24) -> RGB each color has 8 bits
    pixels_2 = image_to_pixel('pengsoo_2.png')
    # <class 'PIL.PngImagePlugin.PngImageFile'>
    # pixels shape (540, 371, 4) # 4 bit -> RGB외에 1가지의 정보 추가 => 투명도 (RGBA) -> 1 byte each (8 bit each)
    # 8x4 = 32 # grey and white grid -> clear bg
    # 각각 모든 grid에 투명도의 값을 줘야하는 것 (0~255로 정의)


    pixel_to_image(pixels_1, 'test1.jpg')
    # 배열에 저장되어져 있는 형태 (jpg, png의 형태가 아닌 배열로 저장되어 있다)
    pixel_to_image(pixels_2, 'test2.png')

    #bmp = bit map picture
    # 파일 크기가 크다 # 16x16 RGBA 파일에서: 16 * 16 * 4
    # 그래서 압축파일들이 생기고 -> png, jpg들이 생성된다
    # header & body 가 있고, header 에 각종 정보 (RGBA, 압축 포맷, 크기, 등등) 이 들어가고, 압축된 사진이 바디로 들어가서 크기가 커도 가능한 것
    # pengsoo.jpg -> test1.jpg의 크기가 다른 이유는 header의 정보가 조금 달라져서
    # 핸드폰 갤러리가 매우 민감민감 (내가 몇월몇일몇시에 어디에 있었다,,까지도 다 나온다) -> 들어갈수도 있고, 안들어갈수도 있다
    # image 라이브러리가 정보를 어떻게 쓰느냐에 따라서 파일의 크기는 달라질 수 있다 (부가적인 정보들의 차이)

    








