# Imports
import gdown
import models as PosterV2
import streamlit as st
import pandas as pd
from main_8 import *
import requests
from PIL import Image
from io import BytesIO

# Download Models
gdown.download(id="17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt", output="models/pretrain/")  
gdown.download(id="1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX", output="models/pretrain/")  
gdown.download(id="1tdYH12vgWnIWfupuBkP3jmWS0pJtDxvh", output="checkpoint/")  

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("GPU available")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

# Model
model = pyramid_trans_expr2(img_size=224, num_classes=8)
model = torch.nn.DataParallel(model).to(device)
checkpoint = torch.load('/mount/src/poster_v2/checkpoint/affectnet-8-model_best.pth',map_location=torch.device(device))
model.load_state_dict(checkpoint['state_dict'])

# Download Image
url = st.text_input('Input your image')

if url:
  data = {
    "Class Labels": [],
    "Percentage": []
  }

  response = requests.get(url)
  image = Image.open(BytesIO(response.content))

  # Transform Image to Tensor
  transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
                                    ])
  image_tensor = transform(image).unsqueeze(0).to(device)

  # Inference
  with torch.no_grad():
      model.eval()
      output = model(image_tensor)

  col1, col2 = st.columns(2)

  st.success('This is a success message!', icon="✅")

  with col1:
    st.subheader("Input Image")
    img = image.resize((512,512))
    st.image(image)
  
  with col2:
    st.subheader("Predictions")
    # Prediction
    class_labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    values, indices = torch.topk(output, len(class_labels))       # Top-8 Predictions
    probabilites = torch.nn.functional.softmax(values)            # Turn Values into Predictions
    percentages = probabilites * 100                              #

    for i in range(len(class_labels)):
      class_index = indices[0][i].item()
      data["Class Labels"].append(class_labels[class_index])
      data["Percentage"].append(f"{percentages[0][i].item():.2f}")
    
    df = pd.DataFrame(data)
    st.dataframe(df)

  






