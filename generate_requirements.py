# generate_requirements.py

required_packages = [
    "streamlit",
    "fastai>=2.7",
    "torch>=1.13",
    "plotly",
    "Pillow"
]

with open("requirements.txt", "w") as f:
    for pkg in required_packages:
        f.write(pkg + "\n")

print("✅ requirements.txt yaratildi.")