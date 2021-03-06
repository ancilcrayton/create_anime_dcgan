import glob
import os
import animeface
from PIL import Image

# Create a data/ directory if it does not exist
if os.path.isdir("data/")==False:
    os.system('mkdir data/')
    
# Track the total number of images with faces
total_num_faces = 0

# Loop over image files scraped using gallery-dl
for index, filename in enumerate(glob.glob('gallery-dl/danbooru/face/*.*')):
    # Open image and detect faces
    try:
        im = Image.open(filename)
        faces = animeface.detect(im)

    except Exception as e:
        print("Exception:{}".format(e))
        continue
        
    # If no faces found in the current image
    if len(faces) == 0:
        print("No faces found in the image")
        continue

    fp = faces[0].face.pos
    
    # Get coordinates of the face detected in the image
    coordinates = (fp.x, fp.y, fp.x+fp.width, fp.y+fp.height)
    
    # Crop image
    cropped_image = im.crop(coordinates)
    
    # Resize image
    cropped_image = cropped_image.resize((64, 64), Image.ANTIALIAS)
    
    # Show cropped and resized image
    # cropped_image.show()
    
    # Save it in the output directory
    cropped_image.save("data/face_{}.png".format(str(index)))
    print("Image cropped and saved successfully!")
    total_num_faces += 1
    print("Number of faces detected till now:{}".format(total_num_faces))

print("Total number of faces:{}".format(total_num_faces))
