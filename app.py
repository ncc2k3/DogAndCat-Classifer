import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet34

# Thiết lập trang web
st.set_page_config(page_title="Dog vs Cat Classifier", layout="wide")

# Tải mô hình
class CustomResNet34(torch.nn.Module):
    def __init__(self, output_classes):
        super(CustomResNet34, self).__init__()
        self.base_model = resnet34(weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = torch.nn.Linear(in_features, output_classes)

    def forward(self, x):
        return self.base_model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomResNet34(output_classes=2).to(device)
model.load_state_dict(torch.load("DogandCat_resnet34.pth", map_location=device))
model.eval()

# Chuẩn bị transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Hàm dự đoán
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return "Cat 🐱" if predicted.item() == 0 else "Dog 🐶"

# Header
st.title("🐾 Dog vs Cat Classifier")
st.subheader("Upload an image to classify whether it's a dog or a cat!")

# Chia bố cục giao diện
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("Upload and Preview")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.header("Prediction Result")
    if uploaded_file is not None:
        label = predict_image(image)
        st.success(f"**Prediction: {label}**")
    else:
        st.info("Please upload an image to see the result.")

# Footer
st.markdown(
    """
    ---
    Developed with ❤️ using **Application**.  
    Model: ResNet34 | Optimized for **Dog vs Cat Classification**.
    """
)
