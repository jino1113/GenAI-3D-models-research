import sys
sys.path.append("./")  

from Hunyuan3D import inference   

result = inference.generate_model(
    image="path_to_image.jpg",
    output_dir="./output"
)
print("Done!", result)
